# -*- coding: utf-8 -*-
"""
Post-fusion acronym / full-form audit for proper-noun entity types.

FAISS blocking's SBERT embeddings routinely miss the lexical relationship
between an acronym and its full expansion (e.g. "ACM JOCCH" vs "ACM Journal
on Computing and Cultural Heritage") -- the two strings don't share enough
surface/semantic content once the shared prefix is stripped, so the pair
never clears --threshold, never becomes a grey-zone candidate row, and
node_fusion.py never sees an edge to merge them on.

This runs AFTER node_fusion.py, scoped to the proper-noun types where that
failure mode actually bites AND where the post-fusion canonical entity
count is small enough to hand an LLM the type's whole roster in one call:
Conference, Journal, Person, Organization, Team. (Task/Method/Dataset/
Metric can number in the thousands post-fusion -- a single-prompt roster
doesn't scale there; out of scope for this script.)

For each type, every canonical entity gets: its label, a few deduped
evidence quotes (same convention faiss_blocking.py's evidence_for uses),
a type-specific discriminating signal already used elsewhere in this
pipeline as a merge guardrail (Conference: confDate/confVenue: Person:
owning email address(es) via hasOwner; Team: affiliation target via the
affiliation relation), and how many raw mentions it already absorbed in the
primary fusion pass (mergedFrom count). The LLM sees the WHOLE roster for
one type at once and groups any ids it believes are the same real-world
entity.

Output: a CSV of candidate MATCH pairs (entity1_id, entity2_id,
entity_type, llm_confidence, reason), same shape as llm_judgement.py's
verified-match rows. This script does NOT merge anything itself -- feed the
output into a second node_fusion-style pass to actually apply the merges.
"""
import argparse
import json
import logging
import os
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=True)

_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "acronym_audit_prompt.txt"

AUDIT_TYPES = {
    'Conference':   'confTitle',
    'Journal':      'journalTitle',
    'Person':       'personName',
    'Organization': 'orgName',
    'Team':         'teamName',
}

MAX_CONTEXT_QUOTES = 3


def setup_logger(log_file):
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )


def _load_properties(df_ent):
    props_by_id, type_by_id = {}, {}
    for _, row in df_ent.iterrows():
        eid = str(row['id']).strip()
        type_by_id[eid] = row['type']
        try:
            props_by_id[eid] = json.loads(row['properties'])
        except Exception:
            props_by_id[eid] = {}
    return props_by_id, type_by_id


def _build_evidence_map(df_rel):
    """entity id -> deduped, capped list of context quotes off its own
    relations -- same convention faiss_blocking.py's evidence_for uses."""
    ctx_map = defaultdict(list)
    for _, r in df_rel.iterrows():
        ctx = str(r.get('context', '')).strip()
        if ctx and ctx.lower() != 'nan':
            ctx_map[r['start_id']].append(ctx)
            ctx_map[r['end_id']].append(ctx)
    evidence = {}
    for eid, quotes in ctx_map.items():
        deduped = list(dict.fromkeys(quotes))[:MAX_CONTEXT_QUOTES]
        evidence[eid] = " | ".join(deduped)
    return evidence


def _build_person_emails(df_rel, props_by_id, type_by_id):
    """Person/Team id -> set of owning email address strings, via hasOwner
    edges (EmailID -[hasOwner]-> Person|Team) and the EmailID's own 'eID'
    property. Mirrors faiss_blocking.py's owner_email_ids, but resolved down
    to the actual address string rather than left as an EmailID node id, so
    it's directly readable in the prompt."""
    owner_emails = defaultdict(set)
    for _, r in df_rel[df_rel['relation'] == 'hasOwner'].iterrows():
        owner_id, target_id = r['start_id'], r['end_id']
        if type_by_id.get(owner_id) == 'EmailID':
            addr = props_by_id.get(owner_id, {}).get('eID')
            if addr:
                owner_emails[target_id].add(addr)
    return owner_emails


def _build_team_affiliations(df_rel, props_by_id, type_by_id):
    """Team id -> set of readable affiliation-target labels (Organization/
    Journal/Conference name), via the 'affiliation' relation. Mirrors
    faiss_blocking.py's team_affiliations, resolved to a label string."""
    label_key_by_type = {'Organization': 'orgName', 'Journal': 'journalTitle', 'Conference': 'confTitle'}
    affiliations = defaultdict(set)
    for _, r in df_rel[df_rel['relation'] == 'affiliation'].iterrows():
        team_id, target_id = r['start_id'], r['end_id']
        target_type = type_by_id.get(target_id)
        label_key = label_key_by_type.get(target_type)
        if label_key:
            label = props_by_id.get(target_id, {}).get(label_key)
            if label:
                affiliations[team_id].add(label)
    return affiliations


def _build_person_cooccurrence(df_rel, type_by_id):
    """Person id -> set of other Person ids that share at least one relation
    TARGET (e.g. both worksOn the same task, both attend the same meeting)
    -- a genuine contextual anchor per Rule 3's own definition ("an
    overlapping project/topic"), but one the LLM has been shown not to
    reconstruct reliably by itself from scattered per-entity evidence quotes
    spread across a 30+ item roster (confirmed case: 'Dr Chatterjee' and
    'Ananya Chatterjee' both worksOn the same task, yet the model never
    proposed them as a group). Computed deterministically here instead of
    hoping the model notices it, then handed to the prompt as an explicit
    fact rather than left implicit in the evidence text."""
    target_to_persons = defaultdict(set)
    for _, r in df_rel.iterrows():
        if type_by_id.get(r['start_id']) == 'Person':
            target_to_persons[r['end_id']].add(r['start_id'])

    cooccur = defaultdict(set)
    for target, persons in target_to_persons.items():
        if len(persons) > 1:
            for p in persons:
                cooccur[p] |= (persons - {p})
    return cooccur


def _merged_from_count(props):
    mf = props.get('mergedFrom')
    if isinstance(mf, list):
        return len(mf)
    return 0


def build_roster(entity_type, df_ent, props_by_id, type_by_id, evidence_map, owner_emails, team_affiliations,
                  person_cooccurrence=None):
    """One dict per canonical entity of `entity_type`, with everything the
    prompt needs to judge it: label, evidence, a type-specific discriminator
    signal, and how many raw mentions it already absorbed."""
    label_key = AUDIT_TYPES[entity_type]
    roster = []
    for _, row in df_ent[df_ent['type'] == entity_type].iterrows():
        eid = str(row['id']).strip()
        props = props_by_id.get(eid, {})
        label = props.get(label_key)
        if not label:
            continue

        discriminator = None
        if entity_type == 'Conference':
            bits = []
            if props.get('confDate'):
                bits.append(f"date={props['confDate']}")
            if props.get('confVenue'):
                bits.append(f"venue={props['confVenue']}")
            discriminator = ", ".join(bits) if bits else None
        elif entity_type == 'Person':
            bits = []
            emails = owner_emails.get(eid)
            if emails:
                bits.append(f"known email(s): {', '.join(sorted(emails))}")
            cooccur = (person_cooccurrence or {}).get(eid)
            if cooccur:
                bits.append(f"co-occurs on the same task/meeting/etc as id(s): {', '.join(sorted(cooccur))}")
            discriminator = "; ".join(bits) if bits else None
        elif entity_type == 'Team':
            affils = team_affiliations.get(eid)
            discriminator = f"affiliated with: {', '.join(sorted(affils))}" if affils else None

        roster.append({
            "id": eid,
            "label": label,
            "evidence": evidence_map.get(eid, ""),
            "discriminator": discriminator,
            "prior_merge_count": _merged_from_count(props),
        })
    return roster


def build_user_prompt(entity_type, roster):
    lines = [f"Entity type: {entity_type}. Roster ({len(roster)} entities):", ""]
    for e in roster:
        parts = [f'id="{e["id"]}"', f'label="{e["label"]}"']
        if e["discriminator"]:
            parts.append(e["discriminator"])
        if e["prior_merge_count"]:
            parts.append(f'already absorbed {e["prior_merge_count"]} raw mention(s)')
        if e["evidence"]:
            parts.append(f'evidence="{e["evidence"]}"')
        lines.append("- " + " | ".join(parts))
    lines.append("")
    lines.append("Output the JSON array now.")
    return "\n".join(lines)


# --- Deterministic post-filter -------------------------------------------- #
# Three rounds of prompt tightening left the model still unreliable at
# holding these rules over a long roster (confirmed: it will name the exact
# violation in its own "reason" text and merge anyway). These checks enforce
# the checkable ones in code instead of hoping the model complies -- same
# pattern faiss_blocking.py already uses for passes_conference_year_check
# etc. Only genuinely checkable facts are enforced here; anything needing
# real judgment (is this really an acronym, is this contextual anchor
# genuine) is still left to the LLM.

YEAR_TOKEN_RE = re.compile(r'(?<!\d)(?:19|20)\d{2}(?!\d)')


def extract_years(text):
    return {int(m.group()) for m in YEAR_TOKEN_RE.finditer(str(text or ''))}


SUBCOMPONENT_KEYWORDS = {
    'office', 'department', 'directorate', 'division', 'branch', 'bureau',
    'secretariat', 'committee', 'chairs', 'proceedings', 'editorial',
    'publications', 'directorate',
}


def is_subcomponent_pair(label1, label2):
    """True if one label is a sub-unit of the other (Rule 5: 'Stanford
    University' vs 'Stanford University Office of Communications') rather
    than an acronym/full-form pair -- one label strictly contains the other
    AND the extra text names an organizational sub-unit."""
    a, b = str(label1).strip().lower(), str(label2).strip().lower()
    if not a or not b or a == b:
        return False
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    if shorter not in longer:
        return False
    remainder = longer.replace(shorter, '', 1)
    return any(kw in remainder for kw in SUBCOMPONENT_KEYWORDS)


_SIG_WORD_STOPWORDS = {
    'the', 'a', 'an', 'of', 'for', 'and', 'in', 'on', 'with', 'to', 'from',
    'at', 'is', 'or', 'university', 'team', 'group', 'office', 'department',
    'committee', 'dr', 'mr', 'mrs', 'ms',
}


def _significant_words(label):
    # Letters only -- a shared numeric token (e.g. both labels containing
    # "2022") is not lexical relatedness and must not count as one; that
    # false positive is exactly what let 'ACL 2022' <-> 'JCDL 2022' (two
    # unrelated series that merely share a year) pass this check.
    words = re.findall(r"[a-z]+", str(label).lower())
    return {w for w in words if len(w) >= 3 and w not in _SIG_WORD_STOPWORDS}


def _initials_of(text):
    return ''.join(w[0] for w in re.findall(r"[A-Za-z]+", str(text)))


def _is_subsequence(needle, haystack):
    it = iter(haystack)
    return all(ch in it for ch in needle)


def _looks_acronym_related(label1, label2):
    a, b = str(label1).strip(), str(label2).strip()
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    short_letters = re.sub(r'[^A-Za-z]', '', shorter).upper()
    if not (2 <= len(short_letters) <= 10):
        return False
    long_compact = re.sub(r'[^A-Za-z]', '', longer).upper()
    if short_letters in long_compact:
        return True
    return _is_subsequence(short_letters, _initials_of(longer).upper())


def has_lexical_relationship(label1, label2):
    """Backstop against wholesale hallucinated merges between entities that
    share nothing at all (e.g. 'press office at IACS' <-> 'Department of
    Computer Science', 'Nanda Roy' <-> 'Laura Jenkins') -- requires either a
    shared significant word or a plausible acronym relationship."""
    return bool(_significant_words(label1) & _significant_words(label2)) \
        or _looks_acronym_related(label1, label2)


def passes_deterministic_checks(entity_type, label1, props1, label2, props2):
    """Hard, code-enforced backstops for the rules the LLM has repeatedly
    failed to hold reliably. Returns (passes: bool, reason: str)."""
    years1 = extract_years(label1) | (extract_years((props1 or {}).get('confDate')) if entity_type == 'Conference' else set())
    years2 = extract_years(label2) | (extract_years((props2 or {}).get('confDate')) if entity_type == 'Conference' else set())
    # Same three-way policy as faiss_blocking.py's passes_conference_year_check:
    # both dated and overlapping -> fine; both dated and disjoint -> reject;
    # EXACTLY ONE side dated -> reject too (a bare/undated label, e.g. "ACL",
    # must never be treated as equivalent to a specific dated instance, e.g.
    # "ACL 2023" -- that's the exact bridge that let unrelated years chain
    # together transitively in earlier runs). Only when NEITHER side has a
    # resolvable year is the pair left to the lexical check below.
    if years1 or years2:
        if not (years1 and years2 and (years1 & years2)):
            return False, f"year mismatch or one side undated ({sorted(years1) or 'none'} vs {sorted(years2) or 'none'})"

    if entity_type in ('Organization', 'Team') and is_subcomponent_pair(label1, label2):
        return False, "sub-component/whole relationship, not the same entity"

    if not has_lexical_relationship(label1, label2):
        return False, "no shared word or acronym relationship between labels"

    return True, ""


def drop_tainted_components(pair_rows):
    """Pairwise checks alone are not enough to enforce the year-conflict
    rule: a bare/no-year label (e.g. "ACL", "ACL Demo track") has nothing
    for the pairwise check to compare, so it passes every individual pair,
    yet still acts as a bridge -- once these pairs are treated as graph
    edges (which the eventual fusion step does, via connected components),
    a bare label sitting between "ACL 2019" and "ACL 2024" silently
    reconnects two entities that were each correctly rejected from
    matching each other directly. Confirmed empirically: this happened for
    ACL/EMNLP/JCDL even after every individual pair passed
    passes_deterministic_checks.

    This closes that gap at the cluster level: build the graph implied by
    the surviving pairs, and for any connected component whose members
    collectively carry 2+ distinct years, drop every pair touching that
    component wholesale -- not just the directly-conflicting ones -- since
    there is no reliable way to know which subset of a tainted component is
    actually safe. Matches the same drop-rather-than-guess stance as the
    rest of the pipeline (a missed merge is cheaper than a false one)."""
    if not pair_rows:
        return pair_rows

    G = nx.Graph()
    for r in pair_rows:
        G.add_edge(r["entity1_label"], r["entity2_label"])

    tainted_labels = set()
    for comp in nx.connected_components(G):
        years_seen = set()
        for label in comp:
            years_seen |= extract_years(label)
        if len(years_seen) > 1:
            tainted_labels |= comp

    if not tainted_labels:
        return pair_rows

    kept, dropped = [], []
    for r in pair_rows:
        if r["entity1_label"] in tainted_labels or r["entity2_label"] in tainted_labels:
            dropped.append(r)
        else:
            kept.append(r)

    if dropped:
        logging.info(f"  Cluster-level year-taint check dropped {len(dropped)} pair(s) spanning "
                     f"{len(tainted_labels)} entities across {sum(1 for c in nx.connected_components(G) if c & tainted_labels)} "
                     f"tainted component(s) -- these passed the per-pair check individually but "
                     f"transitively reconnect conflicting years through a bare/no-year bridge label.")
    return kept


JSON_ARRAY_RE = re.compile(r'\[.*\]', re.DOTALL)


def parse_groups(text, valid_ids, min_confidence=0.5):
    """Best-effort JSON parse of the model's response. Falls back to
    extracting the first [...] span if the model wrapped it in prose or a
    code fence despite instructions. Drops any group referencing an unknown
    id or with fewer than 2 valid ids -- a hallucinated id is a parsing
    failure, not a merge decision, and must not silently propagate.

    Also drops any group whose stated confidence is below min_confidence.
    This matters: despite the prompt asking for ONLY genuine matches, the
    model has been observed emitting groups with confidence 0.0 and a reason
    like "different affiliations" -- i.e. explicitly telling us it's NOT a
    match, just in the same JSON shape as a real one. Without this filter
    those got silently written to --output as llm_prediction=MATCH, which
    is exactly backwards. A missing/non-numeric confidence is NOT dropped
    here (treated as a parsing gap, not a stated rejection)."""
    raw = (text or "").strip()
    raw = re.sub(r'^```(?:json)?|```$', '', raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        m = JSON_ARRAY_RE.search(raw)
        if not m:
            logging.warning(f"Could not parse a JSON array from model output: {raw[:200]!r}")
            return []
        try:
            parsed = json.loads(m.group())
        except Exception:
            logging.warning(f"Could not parse extracted JSON array: {m.group()[:200]!r}")
            return []

    groups, dropped_low_conf = [], 0
    for item in parsed if isinstance(parsed, list) else []:
        ids = [str(i) for i in item.get("ids", []) if str(i) in valid_ids]
        if len(ids) < 2:
            continue
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < min_confidence:
            dropped_low_conf += 1
            continue
        groups.append({
            "ids": ids,
            "confidence": confidence,
            "reason": item.get("reason", ""),
        })
    if dropped_low_conf:
        logging.info(f"Dropped {dropped_low_conf} group(s) with stated confidence below {min_confidence} "
                     f"(the model was explicitly signaling these as non-matches).")
    return groups


def call_openai(sys_prompt, user_prompt, args):
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(base_url=args.base_url, api_key=api_key or "EMPTY") if args.base_url \
        else OpenAI(api_key=api_key)
    is_gpt5 = bool(re.search(r"gpt-5", str(args.model), re.IGNORECASE))
    kwargs = dict(
        model=args.model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if not is_gpt5:
        kwargs["temperature"] = 0.0
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def call_vllm(sys_prompt, user_prompt, args, engine_state):
    """Lazily initializes the vLLM engine on first call and reuses it across
    the (at most 5) type rosters processed in one run -- avoids reloading
    the model per type.

    GPU pinning happens here, not at module import time (unlike
    faiss_blocking.py/llm_judgement.py, which import torch at module level
    and so must parse --gpu out of sys.argv before argparse even runs):
    torch/vllm are only ever imported lazily, inside this function, which
    runs after args.gpu is already known -- so CUDA_VISIBLE_DEVICES can just
    be set from args.gpu directly, before the first import of torch/vllm."""
    if engine_state.get("llm") is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        import torch  # noqa: F401  (import kept local to mirror llm_judgement.py's lazy-import convention)
        from transformers import AutoTokenizer
        from vllm import LLM
        logging.info(f"Pinned to physical GPU {args.gpu} (visible as cuda:0)")
        logging.info("Initializing vLLM engine...")
        engine_state["llm"] = LLM(
            model=args.model, tensor_parallel_size=1, max_model_len=8192,
            gpu_memory_utilization=0.90, enable_prefix_caching=True,
        )
        engine_state["tokenizer"] = AutoTokenizer.from_pretrained(args.model)

    from vllm import SamplingParams
    llm, tokenizer = engine_state["llm"], engine_state["tokenizer"]
    full_prompt = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate([full_prompt], params, use_tqdm=False)
    return outputs[0].outputs[0].text


def main():
    parser = argparse.ArgumentParser(description="Post-fusion acronym/full-form audit (Conference/Journal/Person/Organization/Team)")
    parser.add_argument("--fused_entities",  required=True, help="node_fusion.py output entities CSV")
    parser.add_argument("--fused_relations", required=True, help="node_fusion.py output relations CSV")
    parser.add_argument("--output",          required=True, help="Output candidate-pairs CSV")
    parser.add_argument("--backend",  choices=["openai", "vllm"], default="openai")
    parser.add_argument("--model",    default="gpt-5.1", help="OpenAI model name, or HF model id for --backend vllm")
    parser.add_argument("--base_url", default=None, help="OpenAI-compatible base URL (e.g. local server). Only for --backend openai.")
    parser.add_argument("--gpu",      default="0",
                         help="Physical GPU id to pin for --backend vllm (PCI-bus order; matches "
                              "nvidia-smi). Ignored for --backend openai. Default: 0")
    parser.add_argument("--types",    nargs="+", default=list(AUDIT_TYPES.keys()),
                         help=f"Subset of types to audit (default: all of {list(AUDIT_TYPES.keys())})")
    parser.add_argument("--min_confidence", type=float, default=0.5,
                         help="Drop any returned group whose stated confidence is below this (default: "
                              "0.5). The model has been observed emitting confidence-0.0 groups as its "
                              "way of flagging a considered-but-rejected pair -- those must not be "
                              "written to --output as a MATCH candidate.")
    parser.add_argument("--log", default="acronym_audit.log")
    args = parser.parse_args()

    setup_logger(args.log)
    logging.info("======== ACRONYM / FULL-FORM AUDIT STARTED ========")

    df_ent = pd.read_csv(args.fused_entities)
    df_rel = pd.read_csv(args.fused_relations)
    props_by_id, type_by_id = _load_properties(df_ent)
    evidence_map = _build_evidence_map(df_rel)
    owner_emails = _build_person_emails(df_rel, props_by_id, type_by_id)
    team_affiliations = _build_team_affiliations(df_rel, props_by_id, type_by_id)
    person_cooccurrence = _build_person_cooccurrence(df_rel, type_by_id)

    sys_prompt = _PROMPT_FILE.read_text(encoding="utf-8")
    engine_state = {"llm": None, "tokenizer": None}

    all_rows = []
    for entity_type in args.types:
        if entity_type not in AUDIT_TYPES:
            logging.warning(f"Skipping unknown/out-of-scope type '{entity_type}' (expected one of {list(AUDIT_TYPES.keys())}).")
            continue

        roster = build_roster(entity_type, df_ent, props_by_id, type_by_id, evidence_map, owner_emails, team_affiliations,
                               person_cooccurrence=person_cooccurrence)
        logging.info(f"{entity_type}: {len(roster)} canonical entities in roster.")
        if len(roster) < 2:
            continue

        user_prompt = build_user_prompt(entity_type, roster)
        valid_ids = {e["id"] for e in roster}

        if args.backend == "openai":
            response_text = call_openai(sys_prompt, user_prompt, args)
        else:
            response_text = call_vllm(sys_prompt, user_prompt, args, engine_state)

        groups = parse_groups(response_text, valid_ids, min_confidence=args.min_confidence)
        logging.info(f"{entity_type}: LLM returned {len(groups)} candidate group(s).")

        label_key = AUDIT_TYPES[entity_type]
        n_filtered = 0
        type_rows = []
        for g in groups:
            conf = g["confidence"] if isinstance(g["confidence"], (int, float)) else None
            for id1, id2 in combinations(sorted(g["ids"]), 2):
                props1, props2 = props_by_id.get(id1, {}), props_by_id.get(id2, {})
                label1, label2 = props1.get(label_key), props2.get(label_key)
                ok, veto_reason = passes_deterministic_checks(entity_type, label1, props1, label2, props2)
                if not ok:
                    n_filtered += 1
                    logging.info(f"  Deterministic filter dropped {label1!r} <-> {label2!r}: {veto_reason} "
                                 f"(LLM reason was: {g['reason']!r})")
                    continue
                type_rows.append({
                    "entity_type": entity_type,
                    "entity1_id": id1,
                    "entity1_label": label1,
                    "entity2_id": id2,
                    "entity2_label": label2,
                    "llm_prediction": "MATCH",
                    "llm_confidence": conf,
                    "reason": g["reason"],
                })
        if n_filtered:
            logging.info(f"{entity_type}: deterministic pairwise filter dropped {n_filtered} pair(s).")

        type_rows = drop_tainted_components(type_rows)
        all_rows.extend(type_rows)

    df_out = pd.DataFrame(all_rows, columns=[
        "entity_type", "entity1_id", "entity1_label", "entity2_id", "entity2_label",
        "llm_prediction", "llm_confidence", "reason",
    ])
    df_out.to_csv(args.output, index=False)

    logging.info(f"Total candidate pairs: {len(df_out)}")
    logging.info(f"Output saved to '{args.output}'.")
    logging.info("======== ACRONYM / FULL-FORM AUDIT COMPLETED ========")
    logging.info("NOTE: this script only detects candidate pairs -- it does not merge "
                 "anything. Review --output, then feed it into a second node_fusion-style "
                 "pass to actually apply the merges.")


if __name__ == "__main__":
    main()
