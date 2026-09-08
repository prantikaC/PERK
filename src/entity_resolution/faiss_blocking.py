# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import re
import sys


# --- Pin the GPU BEFORE importing torch ------------------------------------ #
# torch reads CUDA_VISIBLE_DEVICES exactly once, at import time; setting it
# afterwards has no effect. We therefore parse --gpu from argv here, mask all
# other GPUs, and force PCI-bus ordering so --gpu N is PHYSICAL GPU N (the same
# number nvidia-smi shows). After masking, the chosen card is the only visible
# device, so inside the program it is always cuda:0 -- no other GPU can be touched.
def _pin_gpu_from_argv(default="0"):
    gpu = os.environ.get("PERK_GPU", default)
    for i, a in enumerate(sys.argv):
        if a == "--gpu" and i + 1 < len(sys.argv):
            gpu = sys.argv[i + 1]
        elif a.startswith("--gpu="):
            gpu = a.split("=", 1)[1]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return str(gpu)

_PINNED_GPU = _pin_gpu_from_argv()

import faiss
import pandas as pd
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer

TARGET_KEYS = ['datasetName', 'methodName', 'taskName', 'metricName',
               'personName', 'journalTitle', 'confTitle', 'teamName', 'orgName']


def passes_date_check(props1, props2):
    """Reject a pair if any date-suffixed property (taskDate, confDate,
    statusDate, meetDate, mailDate, ... -- PERKOnto's date properties all
    follow this naming convention) is present and non-empty on both sides
    but disagrees. A date missing on one or both sides doesn't block --
    only a confirmed mismatch does. Applies uniformly to every type, not
    just Task, since any entity carrying a date property is exposed to the
    same silent-conflict-drop risk on merge otherwise."""
    date_keys = {k for k in set(props1) | set(props2) if k.lower().endswith('date')}
    for key in date_keys:
        v1, v2 = props1.get(key), props2.get(key)
        if v1 and v2 and v1 != v2:
            return False
    return True


TITLE_YEAR_RE = re.compile(r'\d{2,4}')
ISO_YEAR_RE = re.compile(r'^\s*(\d{4})')


def extract_years_from_title(text):
    """Pull every plausible year out of a title string, normalizing 2-digit
    forms (ACL19, ACL'19) to 4-digit (2019). Deliberately only used for
    Conference, where confTitle reliably encodes a real event year --
    applying this to arbitrary text (e.g. a Dataset named "COVID-19 Corpus")
    would misread an unrelated digit run as a year."""
    years = set()
    for m in TITLE_YEAR_RE.finditer(text or ''):
        s = m.group()
        if len(s) == 4 and s[:2] in ('19', '20'):
            years.add(int(s))
        elif len(s) == 2:
            years.add(2000 + int(s))
    return years


def extract_year_from_date(text):
    """Only the LEADING 4-digit year of an ISO-ish date string ("2019-09-22",
    "2019", "2019-09") -- unlike title text, a date string's other digit
    groups (month, day) are NOT candidate years and must never be scanned
    the way extract_years_from_title does, or "2019-09-22" misreads its "09"
    and "22" groups as years 2009/2022."""
    if not text:
        return set()
    m = ISO_YEAR_RE.match(text)
    if m and m.group(1)[:2] in ('19', '20'):
        return {int(m.group(1))}
    return set()


def passes_conference_year_check(props1, props2):
    """Reject a Conference pair unless both sides agree on year, or NEITHER
    side has any year signal at all:
      - both resolve a year and they overlap ("ACL19" / "ACL 2019" / "ACL"
        with confDate=2019-.. all resolve to {2019})           -> allow
      - both resolve a year and they're disjoint ("ACL 2019" vs "ACL 2020") -> reject
      - exactly one side resolves a year, the other has NONE at all (no
        year in confTitle, no confDate) -- do NOT assume the undated side
        belongs to whichever year the other side has; that's asserting a
        fact with no evidence behind it, not resolving an ambiguity -> reject
      - neither side resolves any year -- genuinely ambiguous on both sides,
        let embedding/LLM judgment decide                                  -> allow
    """
    def years_of(props):
        return extract_years_from_title(props.get('confTitle', '')) | extract_year_from_date(props.get('confDate'))
    y1, y2 = years_of(props1), years_of(props2)
    if y1 and y2:
        return bool(y1 & y2)
    if y1 or y2:
        return False
    return True


MAIL_ID_RE = re.compile(r'Mail ID:\s*(.+)', re.IGNORECASE)
HEADER_BOUNDARY_RE = re.compile(
    r'^(?:From|To|Cc|Bcc|Date|Subject|Thread-?ID|Mail-?ID|Message-?ID):',
    re.IGNORECASE
)
FULL_BODY_MAILNUM_LIMIT = 3


def _extract_body(email_text):
    """Same header-boundary detection as header_signature_parser.py's
    extract_body -- kept local here rather than imported, since this module
    lives in a different package (src/entity_resolution vs src/extraction)."""
    lines = email_text.splitlines()
    body_start_index = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if HEADER_BOUNDARY_RE.match(stripped):
            body_start_index = i + 1
        elif line.startswith((' ', '\t')) and i > 0 and body_start_index == i:
            body_start_index = i + 1
    body = "\n".join(lines[body_start_index:]).strip()
    if not body:
        m = re.search(r'\n\s*\n(.+)', email_text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return body


def build_mailnum_to_body(corpus_path):
    """mailNum -> full raw email body text, parsed from the source corpus
    (same EMAIL_END-delimited format header_signature_parser.py reads)."""
    with open(corpus_path, encoding='utf-8') as f:
        raw_emails = [e.strip() for e in f.read().split('EMAIL_END') if e.strip()]
    mailnum_to_body = {}
    for email_text in raw_emails:
        m = MAIL_ID_RE.search(email_text)
        if m:
            mailnum_to_body[m.group(1).strip()] = _extract_body(email_text)
    return mailnum_to_body


def build_entity_to_mailnums(df_rel):
    """entity id -> set of mailNums, read directly off the 'source' column
    of that entity's own relations (already meaningful mailNums, possibly
    semicolon-joined for relations touching several emails)."""
    entity_mailnums = defaultdict(set)
    for _, r in df_rel.iterrows():
        source = str(r.get('source', '') or '').strip()
        if not source or source.lower() == 'nan':
            continue
        for mn in source.split(';'):
            mn = mn.strip()
            if mn:
                entity_mailnums[r['start_id']].add(mn)
                entity_mailnums[r['end_id']].add(mn)
    return entity_mailnums


EMAIL_OWNER_TYPES = {'Person', 'Team'}


def build_owner_email_ids(df_rel):
    """(Person|Team) entity id -> set of EmailID ids that hasOwner-point to it."""
    owner_email_ids = defaultdict(set)
    for _, r in df_rel[df_rel['relation'] == 'hasOwner'].iterrows():
        owner_email_ids[r['end_id']].add(r['start_id'])
    return owner_email_ids


def passes_owner_email_check(id1, id2, owner_email_ids):
    """Two Person or Team candidates are only allowed to merge if their known
    EmailID ownership overlaps -- if both have at least one known EmailID and
    those sets share none, they're rejected regardless of name-embedding
    similarity (same name, different address, is not enough on its own).
    Missing ownership data on either side doesn't block -- that's an
    incomplete-data case, not a confirmed mismatch."""
    ids1, ids2 = owner_email_ids.get(id1, set()), owner_email_ids.get(id2, set())
    if ids1 and ids2 and not (ids1 & ids2):
        return False
    return True


def build_team_affiliation_ids(df_rel):
    """Team entity id -> set of Organization/Journal/Conference ids it has an
    affiliation relation to."""
    team_affiliations = defaultdict(set)
    for _, r in df_rel[df_rel['relation'] == 'affiliation'].iterrows():
        team_affiliations[r['start_id']].add(r['end_id'])
    return team_affiliations


def passes_team_affiliation_check(id1, id2, team_affiliations):
    """Two Team candidates are only allowed to merge if their known
    affiliations overlap -- generic department names ("Department of
    Computer Science") recur across many different institutions, so two
    Team nodes sharing that name but affiliated with different, known
    Organization/Journal/Conference ids must not merge just because the
    name embedding looks identical. Missing affiliation data on either side
    doesn't block -- only a confirmed, disjoint mismatch does."""
    ids1, ids2 = team_affiliations.get(id1, set()), team_affiliations.get(id2, set())
    if ids1 and ids2 and not (ids1 & ids2):
        return False
    return True


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


def main():
    parser = argparse.ArgumentParser(description="FAISS Blocking Funnel for Entity Resolution")
    parser.add_argument("--entities",   required=True, help="Input entities CSV")
    parser.add_argument("--relations",  required=True, help="Input relations CSV")
    parser.add_argument("--output",     required=True, help="Output grey zone CSV")
    parser.add_argument("--threshold",  type=float, required=True,
                        help="Calibrated similarity floor (e.g. 0.6547)")
    parser.add_argument("--top_k",      type=int, default=10, help="FAISS neighbourhood span (default: 10)")
    parser.add_argument("--model",      default="all-mpnet-base-v2", help="SentenceTransformer model name")
    parser.add_argument("--gpu",        default="0",
                        help="Physical GPU id to pin (PCI-bus order; matches nvidia-smi). "
                             "The chosen card is the only one made visible. Default: 0")
    parser.add_argument("--log",        default="pipeline_step1.log")
    parser.add_argument("--corpus",     default=None,
                        help="Optional path to the raw EMAIL_END-delimited corpus (e.g. PATRA.txt). "
                             "When given, an entity tied to a small number of source emails "
                             f"(<= {FULL_BODY_MAILNUM_LIMIT}) is embedded using those emails' full "
                             "bodies instead of just the short per-relation context quotes.")
    args = parser.parse_args()

    setup_logger(args.log)
    logging.info(f"Pinned to physical GPU {_PINNED_GPU} (visible as cuda:0)")
    logging.info(f"Starting FAISS Funnel | threshold={args.threshold} | top_k={args.top_k}")

    df_ent = pd.read_csv(args.entities)
    df_rel = pd.read_csv(args.relations)

    df_rel['start_id'] = df_rel['start_id'].astype(str).str.strip()
    df_rel['end_id']   = df_rel['end_id'].astype(str).str.strip()
    valid_ids = set(df_rel['start_id']).union(set(df_rel['end_id']))
    owner_email_ids = build_owner_email_ids(df_rel)
    team_affiliations = build_team_affiliation_ids(df_rel)

    parsed = []
    for _, row in df_ent.iterrows():
        eid = str(row['id']).strip()
        if eid not in valid_ids:
            continue
        try:
            props = json.loads(row['properties'])
        except Exception:
            continue
        for k in TARGET_KEYS:
            if k in props:
                parsed.append({"id": eid, "type": row['type'], "label": str(props[k]), "raw_props": props})
                break

    df_parsed = pd.DataFrame(parsed)
    logging.info(f"Retained {len(df_parsed)} connected entities matching target ontology.")

    ctx_map = defaultdict(list)
    for _, r in df_rel.iterrows():
        ctx = str(r['context']).strip()
        if ctx and ctx.lower() != "nan":
            ctx_map[r['start_id']].append(ctx)
            ctx_map[r['end_id']].append(ctx)

    mailnum_to_body = build_mailnum_to_body(args.corpus) if args.corpus else {}
    entity_to_mailnums = build_entity_to_mailnums(df_rel) if mailnum_to_body else {}
    full_body_used = 0

    MAX_CONTEXT_QUOTES = 3

    def evidence_for(entity_id):
        mailnums = entity_to_mailnums.get(entity_id, set())
        if mailnum_to_body and 0 < len(mailnums) <= FULL_BODY_MAILNUM_LIMIT:
            bodies = [mailnum_to_body[mn] for mn in mailnums if mn in mailnum_to_body]
            if bodies:
                nonlocal full_body_used
                full_body_used += 1
                return " | ".join(bodies)
        # Hub entities (a Person/Team mentioned across hundreds/thousands of
        # relations) would otherwise join EVERY one of their context quotes
        # unbounded -- one real case joined 4,153 quotes into a single
        # 471,188-character evidence string, both blowing up the embedding
        # signal (drowning the entity's own name in unrelated text) and,
        # far more expensively, getting sent verbatim to the LLM on every
        # candidate pair involving that entity. Dedupe (many relations reuse
        # the same quote) and cap to the first few distinct ones.
        quotes = ctx_map.get(entity_id, ["Missing"])
        deduped = list(dict.fromkeys(quotes))[:MAX_CONTEXT_QUOTES]
        return " | ".join(deduped)

    df_parsed['context'] = df_parsed['id'].apply(evidence_for)
    df_parsed['text_to_embed'] = df_parsed.apply(
        lambda x: f"Entity: {x['label']}. Evidence: {x['context']}", axis=1
    )
    if mailnum_to_body:
        logging.info(f"Full email body used as evidence for {full_body_used} / {len(df_parsed)} entities "
                     f"(<= {FULL_BODY_MAILNUM_LIMIT} source mailNums); short context quotes used for the rest.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading embedding model '{args.model}' to {device}...")
    model = SentenceTransformer(args.model, device=device)

    all_pairs, seen = [], set()
    strict_rejections = threshold_rejections = 0

    for e_type in df_parsed['type'].unique():
        df_t = df_parsed[df_parsed['type'] == e_type].reset_index(drop=True)
        if len(df_t) < 2:
            continue

        embeddings = model.encode(
            df_t['text_to_embed'].tolist(),
            batch_size=64, convert_to_numpy=True, normalize_embeddings=True
        )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        sims, idxs = index.search(embeddings, min(args.top_k, len(df_t)))

        for i in range(len(df_t)):
            for rank in range(1, min(args.top_k, len(df_t))):
                sim = float(sims[i][rank])
                if sim <= args.threshold:
                    threshold_rejections += 1
                    break

                j = idxs[i][rank]
                if i == j:
                    continue

                p_id = tuple(sorted([df_t.iloc[i]['id'], df_t.iloc[j]['id']]))
                if p_id in seen:
                    continue

                if not passes_date_check(df_t.iloc[i]['raw_props'], df_t.iloc[j]['raw_props']):
                    strict_rejections += 1
                    continue

                if e_type in EMAIL_OWNER_TYPES and not passes_owner_email_check(
                    df_t.iloc[i]['id'], df_t.iloc[j]['id'], owner_email_ids
                ):
                    strict_rejections += 1
                    continue

                if e_type == 'Conference' and not passes_conference_year_check(
                    df_t.iloc[i]['raw_props'], df_t.iloc[j]['raw_props']
                ):
                    strict_rejections += 1
                    continue

                if e_type == 'Team' and not passes_team_affiliation_check(
                    df_t.iloc[i]['id'], df_t.iloc[j]['id'], team_affiliations
                ):
                    strict_rejections += 1
                    continue

                seen.add(p_id)
                all_pairs.append({
                    "entity_type":    e_type,
                    "similarity_score": sim,
                    "entity1_id":     df_t.iloc[i]['id'],
                    "entity_label_1": df_t.iloc[i]['label'],
                    "evidence_1":     df_t.iloc[i]['context'],
                    "entity2_id":     df_t.iloc[j]['id'],
                    "entity_label_2": df_t.iloc[j]['label'],
                    "evidence_2":     df_t.iloc[j]['context'],
                })

    df_grey = pd.DataFrame(all_pairs)
    df_grey.to_csv(args.output, index=False)

    logging.info("--- FAISS Funnel Results ---")
    logging.info(f"Strict metadata rejections : {strict_rejections}")
    logging.info(f"Threshold rejections       : {threshold_rejections} (estimated)")
    logging.info(f"Grey zone candidates saved : {len(df_grey)}")
    logging.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
