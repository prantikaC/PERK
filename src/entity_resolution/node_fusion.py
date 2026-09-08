import argparse
import json
import logging
import re
from collections import defaultdict

import networkx as nx
import pandas as pd

TITLE_WORDS = {
    'prof', 'prof.', 'dr', 'dr.', 'mr', 'mrs', 'ms', 'miss',
    'office', 'press', 'editorial', 'program', 'centre', 'center',
    'department', 'institute', 'university', 'college', 'school',
    'committee', 'board', 'chair', 'team', 'group', 'staff',
}


def union_sources(values):
    """When two relations collapse into one duplicate triple after id
    remapping, don't just drop the loser and its 'source' mailNum(s) --
    union every distinct mailNum across all of them (semicolon-joined, same
    convention header_signature_parser.py already uses for a relation
    corroborated by several emails) onto the surviving row."""
    seen = []
    for v in values:
        for part in str(v).split(';'):
            part = part.strip()
            if part and part.lower() not in ('nan', '') and part not in seen:
                seen.append(part)
    return ';'.join(seen)


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


def _as_text(value) -> str:
    """A property value can now be a list (a genuine merge conflict kept
    every variant instead of dropping one) -- take the first/primary variant
    for heuristics that need a single string, rather than crashing on it."""
    if isinstance(value, list):
        return value[0] if value else ''
    return value or ''


def _names_overlap(name_a: str, name_b: str) -> bool:
    """True if the two person name strings are likely the same person."""
    a, b = _as_text(name_a).lower().strip(), _as_text(name_b).lower().strip()
    if not a or not b or len(b) < 4:
        return False
    if a in b or b in a:
        return True
    words_a = {w for w in re.split(r'\s+', a) if len(w) >= 4 and w not in TITLE_WORDS}
    words_b = {w for w in re.split(r'\s+', b) if len(w) >= 4 and w not in TITLE_WORDS}
    return bool(words_a & words_b)


_EMPTY = (None, "", "null", "None", "nan")


def completeness(props: dict) -> int:
    """Count of real (non-empty) property values -- used to pick which
    cluster member becomes canonical, instead of an arbitrary id sort."""
    return sum(1 for v in props.values() if v not in _EMPTY)


# Every property faiss_blocking.py's TARGET_KEYS matches entities on --
# these get abbreviation-aware collapse (see collapse_name_variants) rather
# than plain longest-wins, since an abbreviation/full-expansion pair
# ("IACS" / "Indian Association for the Cultivation of Science", "NLP" /
# "Natural Language Processing") can show up for any of them, not just
# Journal/Conference titles.
NAME_KEYS = {'datasetName', 'methodName', 'taskName', 'metricName',
             'personName', 'teamName', 'orgName', 'journalTitle', 'confTitle'}


def merge_scalar_longest(can_v, obs_v):
    """The default rule for every property except NAME_KEYS (which get
    abbreviation-aware handling below): never produce a list -- QA's
    cypher_prompt filters properties with a bare toLower(x.prop) CONTAINS
    ..., which throws on a list-valued property, and a merge decision has
    already been made by this point (FAISS + LLM judged the pair a match on
    their primary identity property) -- a property that still disagrees
    afterward (e.g. confVenue "Kolkata" vs "Kolkata, West Bengal") isn't
    grounds to second-guess that decision in code; keep the longer, more
    complete value, since a fuller value almost always contains the shorter
    one as a substring rather than contradicting it. A genuine contradiction
    (e.g. "Kolkata" vs an unrelated "Cambridge") reflects a bad match
    decision -- catching that belongs to LLM judgment itself (which sees
    full context before deciding MATCH/NO_MATCH), not to a property-merge
    rule after the fact."""
    if obs_v in _EMPTY:
        return can_v
    if can_v in _EMPTY:
        return obs_v
    return can_v if len(can_v) >= len(obs_v) else obs_v


def collapse_name_variants(values):
    """NAME_KEYS: converge every distinct variant seen across a whole merge
    cluster (full name, abbreviation, partial forms -- e.g. "ACM JOCCH",
    "JOCCH", "ACM Journal on Computing and Cultural Heritage", "IACS" /
    "Indian Association for the Cultivation of Science", "NLP" / "Natural
    Language Processing") onto ONE string, computed once over the FULL set
    rather than by folding pairs one at a time -- pairwise folding
    double-appends the abbreviation once the running value already contains
    an earlier appended form. Take the overall longest variant as the base;
    only if the overall SHORTEST variant (the real abbreviation) isn't
    already a substring of it, append that one in parens. Middle-length
    variants are ignored for the append decision -- they're already close
    enough to the full name not to need separate mention, and the actual
    query term worth guaranteeing CONTAINS-findability for is the bare
    abbreviation. Never a list."""
    values = list(dict.fromkeys(v for v in values if v not in _EMPTY))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    longest = max(values, key=len)
    abbrev = min(values, key=len)
    if abbrev.lower() in longest.lower():
        return longest
    return f"{longest} ({abbrev})"


def accumulate_name_variant(can_v, obs_v):
    """NAME_KEYS-only internal accumulator, used solely as scratch state
    while folding a cluster's members one pair at a time. Never written to
    any output -- collapse_name_variants always runs afterward, over the
    complete accumulated list, and replaces it with a single string before
    df_ent is ever saved. This is what lets collapse_name_variants see
    every distinct variant in the cluster at once instead of two at a time
    (which double-appends the abbreviation -- see collapse_name_variants)."""
    if obs_v in _EMPTY:
        return can_v
    if can_v in _EMPTY:
        return obs_v
    can_list = can_v if isinstance(can_v, list) else [can_v]
    if obs_v in can_list:
        return can_v
    return can_list + [obs_v]


def merge_properties(obs_p: dict, can_p: dict) -> dict:
    """Per-pair fold during cluster resolution. Every property ends up a
    plain scalar (longest-wins) except NAME_KEYS, which accumulate as a
    transient list here and get collapsed to one string in a single pass
    after the whole cluster is folded -- see accumulate_name_variant and
    collapse_name_variants."""
    merged = {}
    for k in set(obs_p) | set(can_p):
        can_v, obs_v = can_p.get(k), obs_p.get(k)
        if k in NAME_KEYS:
            merged[k] = accumulate_name_variant(can_v, obs_v)
        else:
            merged[k] = merge_scalar_longest(can_v, obs_v)
    return merged


def patch_person_properties(df_ent: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Fill missing affiliation on Person nodes that have an email but no affiliation,
    by finding a complementary Person node (same partial name, has affiliation, no email).
    Operates entirely on the entities DataFrame — no Neo4j required.
    """
    persons = df_ent[df_ent['type'] == 'Person'].copy()
    if persons.empty:
        return df_ent, 0

    parsed = {}
    for _, row in persons.iterrows():
        try:
            parsed[row['id']] = json.loads(row['properties'])
        except Exception:
            parsed[row['id']] = {}

    has_email = {eid: p for eid, p in parsed.items() if p.get('personEmail')}
    has_affil = {eid: p for eid, p in parsed.items()
                 if p.get('affiliation') and not p.get('personEmail')}

    patches = {}
    for eid_a, props_a in has_email.items():
        if props_a.get('affiliation'):
            continue
        name_a = props_a.get('personName', '')
        for eid_b, props_b in has_affil.items():
            name_b = props_b.get('personName', '')
            if _names_overlap(name_a, name_b):
                patches[eid_a] = props_b['affiliation']
                break

    if not patches:
        return df_ent, 0

    df_out = df_ent.copy()
    for idx, row in df_out.iterrows():
        if row['id'] in patches:
            try:
                props = json.loads(row['properties'])
                props['affiliation'] = patches[row['id']]
                df_out.at[idx, 'properties'] = json.dumps(props)
            except Exception:
                pass

    return df_out, len(patches)


_TITLE_STOPWORDS = {
    'a', 'an', 'the', 'of', 'for', 'and', 'in', 'on', 'with', 'to', 'from',
    'across', 'into', 'over', 'under', 'via', 'as', 'is', 'at', 'by', 'or',
}


def _significant_title_words(title):
    """Lowercased, stopword-free word set for a paper title -- used only to
    detect two clearly DIFFERENT real papers that happen to share a
    SubmissionID identifier (a corpus/extraction data error, not something
    a title-similarity check can fix), as opposed to the same paper's title
    genuinely being reworded across drafts (which should still share most
    of its substantive vocabulary)."""
    words = re.findall(r"[a-z0-9]+", str(title).lower())
    return {w for w in words if w not in _TITLE_STOPWORDS}


def _title_groups_share_identity(paper_to_words, min_overlap=0.3):
    """Given {paper_id: significant_word_set} for one identifier group,
    partition papers into title-similarity clusters (union by pairwise
    Jaccard overlap >= min_overlap) and return the SET of paper ids in the
    LARGEST cluster. Papers with no resolvable title are kept (can't be
    proven contradictory) but never anchor a cluster on their own."""
    ids = [p for p, words in paper_to_words.items() if words]
    untitled = [p for p, words in paper_to_words.items() if not words]
    if len(ids) <= 1:
        return set(ids) | set(untitled)

    parent = {p: p for p in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            wa, wb = paper_to_words[a], paper_to_words[b]
            overlap = len(wa & wb) / len(wa | wb) if (wa | wb) else 0
            if overlap >= min_overlap:
                union(a, b)

    clusters = defaultdict(set)
    for p in ids:
        clusters[find(p)].add(p)
    largest = max(clusters.values(), key=len)
    return largest | set(untitled)


def build_submission_paper_pairs(df_ent, df_rel):
    """Papers that share the same SubmissionID identifier are the same real
    submission -- merge them deterministically, bypassing FAISS/LLM
    title-similarity entirely. This matters specifically because Paper is
    deliberately excluded from TARGET_KEYS (title-similarity is too risky
    for papers -- distinct papers from the same lab/project routinely share
    most of their title's vocabulary), so without this, duplicate Paper
    nodes for the same submission would never converge. SubmissionID nodes
    themselves are never deduplicated either (not in TARGET_KEYS), so the
    same real submission can easily show up as two different SubmissionID
    entities with the identical identifier string, each pointing at its own
    duplicate Paper -- group on the identifier STRING, not the SubmissionID
    node id, to catch that.

    A Paper that itself has >1 DISTINCT identifier string across its own
    identifies edges is excluded entirely rather than trusted as a bridge --
    confirmed in real data (4/54 Papers in the openai run) that this happens
    from pre-existing body-extraction errors (a generically-titled Paper
    reused across several actually-different submissions). Treating such a
    Paper as a normal member of every identifier group it touches would
    transitively fuse all of those otherwise-unrelated groups into one
    giant, wrong merge via connected components -- confirmed empirically:
    one such Paper turned what should have been several small 2-3-paper
    merges into a single 23-paper one."""
    sub_identifier = {}
    for _, r in df_ent[df_ent['type'] == 'SubmissionID'].iterrows():
        try:
            ident = json.loads(r['properties']).get('identifier')
        except Exception:
            ident = None
        if ident:
            sub_identifier[r['id']] = ident.strip().lower()

    paper_to_idents = defaultdict(set)
    for _, r in df_rel[df_rel['relation'] == 'identifies'].iterrows():
        ident = sub_identifier.get(r['start_id'])
        if ident:
            paper_to_idents[r['end_id']].add(ident)
    contradictory_papers = {p for p, idents in paper_to_idents.items() if len(idents) > 1}
    if contradictory_papers:
        logging.info(f"Excluded {len(contradictory_papers)} Papers with contradictory SubmissionID "
                      f"identifiers from deterministic merging: {sorted(contradictory_papers)}")

    identifier_to_papers = defaultdict(set)
    for _, r in df_rel[df_rel['relation'] == 'identifies'].iterrows():
        ident = sub_identifier.get(r['start_id'])
        if ident and r['end_id'] not in contradictory_papers:
            identifier_to_papers[ident].add(r['end_id'])

    # Reverse-direction check to the contradictory_papers one above: that
    # catches one PAPER claiming multiple identifiers; this catches one
    # IDENTIFIER shared by multiple genuinely different papers -- confirmed
    # real case where a corpus/extraction error reused the same SubmissionID
    # identifier across two entirely unrelated manuscripts ("Domain
    # Adaptation and Evaluation Metrics for Historical NER..." and "NLP
    # Approaches to Historical Entity Disambiguation..."), which would
    # otherwise get silently deterministically merged (mergeConfidence 1.0)
    # into one wrong Paper. Same-paper title drift across drafts still
    # shares most of its substantive vocabulary and won't trigger this;
    # only a near-zero-overlap title pair does.
    paper_title = {}
    for _, row in df_ent[df_ent['type'] == 'Paper'].iterrows():
        try:
            paper_title[row['id']] = json.loads(row['properties']).get('paperTitle', '')
        except Exception:
            paper_title[row['id']] = ''

    pairs = []
    for ident, papers in identifier_to_papers.items():
        paper_to_words = {p: _significant_title_words(paper_title.get(p, '')) for p in papers}
        kept = _title_groups_share_identity(paper_to_words)
        excluded = papers - kept
        if excluded:
            logging.info(
                f"Identifier '{ident}' claimed by papers with unrelated titles -- "
                f"keeping {sorted(kept)}, excluding {sorted(excluded)} from this "
                f"deterministic merge as a likely identifier collision, not the same submission."
            )
        papers = sorted(kept)
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                pairs.append((papers[i], papers[j]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Node Fusion")
    parser.add_argument("--llm_resolved",    required=True, help="Input LLM resolved CSV")
    parser.add_argument("--raw_entities",    required=True, help="Input raw entities CSV")
    parser.add_argument("--raw_relations",   required=True, help="Input raw relations CSV")
    parser.add_argument("--fused_entities",  required=True, help="Output fused entities CSV")
    parser.add_argument("--fused_relations", required=True, help="Output fused relations CSV")
    parser.add_argument("--log", default="pipeline_step3.log", help="Log file path")
    parser.add_argument("--llm_confidence_floor", type=float, default=0.95,
                        help="Minimum llm_confidence to keep a MATCH verdict (default: 0.95, "
                             "calibrated for GPT-5.1 -- see openai_v2/llm_confidence_calibration_log.txt). "
                             "Re-calibrate per model via calibrate_llm_confidence.py before trusting a "
                             "different value; e.g. Qwen2.5-32B's confidence carries no discriminating "
                             "signal in this corpus (no floor beats its raw ~85% precision), so pass "
                             "--llm_confidence_floor 0.0 for a Qwen-judged run instead of this default.")
    args = parser.parse_args()

    setup_logger(args.log)
    logging.info("Starting Graph Node Fusion (Transitive Closure)...")

    df_res = pd.read_csv(args.llm_resolved)
    df_matches = df_res[df_res['llm_prediction'] == 'MATCH']
    logging.info(f"Loaded {len(df_matches)} verified matches for fusion.")

    # See --llm_confidence_floor help: default 0.95 is calibrated for GPT-5.1
    # specifically (openai_v2/llm_confidence_calibration_log.txt) -- pass a
    # different value (or 0.0 to disable) for any other judge model.
    if 'llm_confidence' in df_matches.columns:
        before = len(df_matches)
        low_confidence = df_matches['llm_confidence'].notna() & (df_matches['llm_confidence'] < args.llm_confidence_floor)
        df_matches = df_matches[~low_confidence]
        logging.info(
            f"Dropped {before - len(df_matches)} MATCH verdicts below llm_confidence "
            f"floor {args.llm_confidence_floor} ({len(df_matches)} remain)."
        )

    df_ent = pd.read_csv(args.raw_entities)
    initial_nodes = len(df_ent)
    df_ent.set_index('id', drop=False, inplace=True)

    entity_completeness = {}
    for eid in df_ent.index:
        try:
            entity_completeness[eid] = completeness(json.loads(df_ent.at[eid, 'properties']))
        except Exception:
            entity_completeness[eid] = 0

    df_rel = pd.read_csv(args.raw_relations)
    initial_edges = len(df_rel)

    has_llm_confidence = 'llm_confidence' in df_matches.columns

    G = nx.Graph()
    for _, r in df_matches.iterrows():
        # mergeConfidence should reflect both signals that led to this merge,
        # not just the FAISS embedding similarity that got the pair proposed
        # in the first place -- the LLM's own stated confidence in its
        # MATCH verdict is at least as informative. Average the two when
        # llm_confidence is available; fall back to similarity alone for an
        # older llm_resolved.csv generated before this column existed.
        weight = r['similarity_score']
        if has_llm_confidence and pd.notna(r.get('llm_confidence')):
            weight = (r['similarity_score'] + r['llm_confidence']) / 2
        G.add_edge(r['entity1_id'], r['entity2_id'], similarity=weight)

    submission_pairs = build_submission_paper_pairs(df_ent.reset_index(drop=True), df_rel)
    for a, b in submission_pairs:
        # Deterministic (same SubmissionID), not a FAISS/LLM judgment -- give
        # it a perfect confidence rather than leaving it out of the average.
        G.add_edge(a, b, similarity=1.0)
    logging.info(f"Added {len(submission_pairs)} deterministic Paper merge pairs (shared SubmissionID).")

    id_map = {}
    merged_from = defaultdict(list)
    merge_confidence = {}
    clusters_found = 0
    for cluster in nx.connected_components(G):
        clusters_found += 1
        c_list = sorted(cluster)
        # Canonical = most complete entity in the cluster (most real property
        # values), not an arbitrary lexicographic id pick; ties broken by id
        # for determinism.
        canonical = min(c_list, key=lambda eid: (-entity_completeness.get(eid, 0), eid))
        obs_list = [eid for eid in c_list if eid != canonical]
        for obs in obs_list:
            id_map[obs] = canonical
        merged_from[canonical] = obs_list

        sims = [d['similarity'] for _, _, d in G.subgraph(c_list).edges(data=True)]
        merge_confidence[canonical] = round(sum(sims) / len(sims), 4) if sims else None

    logging.info(f"Graph resolved into {clusters_found} distinct entity clusters.")
    logging.info(f"Identified {len(id_map)} redundant nodes to be merged into canonical IDs.")

    logging.info("Rewriting Relations (Edges)...")
    df_rel['start_id'] = df_rel['start_id'].apply(lambda x: id_map.get(x, x))
    df_rel['end_id']   = df_rel['end_id'].apply(lambda x: id_map.get(x, x))
    df_rel = df_rel[df_rel['start_id'] != df_rel['end_id']]
    df_rel['source'] = df_rel.groupby(
        ['start_id', 'end_id', 'relation', 'context']
    )['source'].transform(union_sources)
    df_rel = df_rel.drop_duplicates(subset=['start_id', 'end_id', 'relation', 'context'])
    final_edges = len(df_rel)
    df_rel.to_csv(args.fused_relations, index=False)

    logging.info("Rewriting Entities (Nodes)...")

    properties_merged = 0
    for obs, can in id_map.items():
        if obs in df_ent.index and can in df_ent.index:
            try:
                obs_p = json.loads(df_ent.at[obs, 'properties'])
                can_p = json.loads(df_ent.at[can, 'properties'])
                merged_p = merge_properties(obs_p, can_p)
                df_ent.at[can, 'properties'] = json.dumps(merged_p)
                properties_merged += 1
            except Exception:
                pass

    # Provenance: every canonical entity that absorbed at least one other
    # node records exactly which raw ids were folded into it, and the
    # average FAISS similarity across the match edges that justified the
    # merge -- so a merge is auditable after the fact, not just a silent
    # deletion of the losing id.
    for can, obs_list in merged_from.items():
        if can not in df_ent.index or not obs_list:
            continue
        try:
            props = json.loads(df_ent.at[can, 'properties'])
        except Exception:
            props = {}
        props['mergedFrom'] = obs_list
        if merge_confidence.get(can) is not None:
            props['mergeConfidence'] = merge_confidence[can]

        # Collapse every NAME_KEYS property down to one string now that all
        # its variants across the cluster have been collected (see merge_properties).
        for key in NAME_KEYS:
            if isinstance(props.get(key), list):
                props[key] = collapse_name_variants(props[key])

        df_ent.at[can, 'properties'] = json.dumps(props)

    df_ent = df_ent[~df_ent['id'].isin(id_map.keys())]
    final_nodes = len(df_ent)

    df_ent, persons_patched = patch_person_properties(df_ent)

    df_ent.to_csv(args.fused_entities, index=False)

    logging.info("--- Node Fusion Results ---")
    logging.info(f"Nodes before fusion:  {initial_nodes}")
    logging.info(f"Nodes after fusion:   {final_nodes} (Removed {initial_nodes - final_nodes})")
    logging.info(f"Properties merged:    {properties_merged}")
    logging.info(f"Edges before fusion:  {initial_edges}")
    logging.info(f"Edges after fusion:   {final_edges} (Consolidated {initial_edges - final_edges})")
    logging.info(f"Person nodes patched: {persons_patched} (affiliation filled from fragment)")
    logging.info(f"Output saved to '{args.fused_entities}' and '{args.fused_relations}'.")


if __name__ == "__main__":
    main()
