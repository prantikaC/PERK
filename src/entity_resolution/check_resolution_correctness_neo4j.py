# -*- coding: utf-8 -*-
"""
Check annotated MATCH/NO_MATCH entity pairs directly against the LIVE Neo4j
graph (not the intermediate CSVs):
  - NO_MATCH pair -> the two entities must exist as separate nodes.
  - MATCH pair    -> the two entities must NOT exist as separate nodes
                      (i.e. resolve to the same node).

For each side of a pair, first tries a direct name match: node(s) of that
:LABEL whose type-specific name property (personName, taskName, ...)
case/whitespace-insensitively matches the annotated label text.

If that finds nothing -- which happens whenever the annotated text was a
mention that got merged away, and the surviving node kept a *different*
mention's text as its name -- falls back to mergedFrom: every node in the
live graph carries its own `mergedFrom` property (the raw pre-fusion ids
absorbed into it, read live off Neo4j, not a CSV snapshot). The annotated
label is looked up by exact text match against the PRE-FUSION extraction
CSV (the only place original per-mention text still exists) to get a raw
id, which is then matched against every live node's mergedFrom list to
find which final node it landed in.

If that STILL finds nothing, and the type is one where two independent
extractions (e.g. GPT vs Qwen) should describe the same real mention
nearly identically -- Person, Conference, Journal, Paper, SubmissionID,
unlike Task/Method/Metric/Dataset which are free-form paraphrases -- tries
a token-containment fuzzy match: honorifics stripped, punctuation-
insensitive, one label's token set a subset of the other's (e.g. "Dr.
Patel" inside "Dr. Ajay Patel"; "ACL 2024" inside "...(ACL 2024)"). For
Conference specifically, this is additionally gated on year agreement --
if both titles carry a 4-digit year, they must match, so "ACL 2019" can
never fuzzy-match "ACL 2024" just because "ACL" is a token subset of both.

Two sides are "the same node" if their resulting id sets intersect. If
nothing finds a match, the pair is UNRESOLVED (can't judge correctness)
rather than silently counted either way -- this is intentional for
genuinely ambiguous cases (e.g. two differently-worded meeting/session
descriptions that MIGHT be the same event) that a token/fuzzy heuristic
shouldn't be trusted to auto-resolve.

Usage:
    python check_resolution_correctness_neo4j.py \
        --pairs balanced_entity_all_pairs_50.csv \
        --raw_entities openai_v2_entities_final.csv \
        --model GPT_V5 \
        --output correctness_report_neo4j.csv
"""

import argparse
import ast
import json
import os
import re
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)

# Types where two independent extractions of the same corpus should describe
# a given real mention almost identically (a name, a code, a title) --
# unlike Task/Method/Metric/Dataset, which are free-form paraphrases and
# where token-containment would be more likely to wrongly conflate two
# genuinely different things.
FUZZY_TYPES = {"Person", "Conference", "Journal", "Paper", "SubmissionID"}

HONORIFIC_RE = re.compile(r"^(?:dr|prof|mr|mrs|ms|miss)\.?\s+", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def strip_honorific(s):
    return HONORIFIC_RE.sub("", s.strip())


def tokenize(s):
    s = strip_honorific(norm(s))
    s = re.sub(r"[^\w\s]", " ", s)
    return {t for t in s.split() if t}


def years_of(s):
    return set(YEAR_RE.findall(s))


def token_containment_match(etype, label_a, label_b):
    ta, tb = tokenize(label_a), tokenize(label_b)
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if not shorter or not shorter.issubset(longer):
        return False
    if etype == "Conference":
        ya, yb = years_of(label_a), years_of(label_b)
        if ya and yb and not (ya & yb):
            return False
    return True

TYPE_LABEL_KEY = {
    "Email": "mailNum",
    "MailThread": "threadID",
    "EmailID": "eID",
    "Person": "personName",
    "Team": "teamName",
    "Organization": "orgName",
    "Paper": "paperTitle",
    "Conference": "confTitle",
    "Journal": "journalTitle",
    "SubmissionID": "identifier",
    "PaperStatus": "statusType",
    "Meeting": "meetAgenda",
    "Dataset": "datasetName",
    "Method": "methodName",
    "Task": "taskName",
    "Metric": "metricName",
}


def get_driver(prefix):
    prefix = prefix.upper()
    uri = os.getenv(f"{prefix}_NEO4J_URI")
    user = os.getenv(f"{prefix}_NEO4J_USERNAME")
    pwd = os.getenv(f"{prefix}_NEO4J_PASSWORD")
    if not all([uri, user, pwd]):
        raise ValueError(f"Missing {prefix}_NEO4J_URI/USERNAME/PASSWORD in .env")
    return GraphDatabase.driver(uri, auth=(user, pwd))


def type_col(row, n):
    for candidate in (f"entity{n}_type", "entity_type"):
        if candidate in row and pd.notna(row[candidate]):
            return row[candidate]
    return None


QUOTE_CHARS = "'\"‘’“”"


def norm(s):
    return str(s).strip().strip(QUOTE_CHARS).strip().lower()


def load_raw_label_index(raw_entities_path):
    """(type, normalized label) -> list of raw pre-fusion entity ids sharing that label."""
    df = pd.read_csv(raw_entities_path)
    index = defaultdict(list)
    for _, row in df.iterrows():
        etype = row["type"]
        key = TYPE_LABEL_KEY.get(etype)
        if not key:
            continue
        try:
            props = json.loads(row["properties"])
        except Exception:
            continue
        label = props.get(key)
        if label is None:
            continue
        index[(etype, norm(label))].append(str(row["id"]).strip())
    return index


def load_live_merged_from_map(session):
    """live final node id -> parsed mergedFrom list of raw pre-fusion ids it absorbed.

    Queries EVERY node, not just ones with mergedFrom set -- an untouched
    node that nothing was ever merged into still needs a trivial id entry,
    or its own raw id (== its final id) never appears anywhere in the map.
    """
    rows = session.run("MATCH (n) WHERE n.id IS NOT NULL "
                        "RETURN n.id AS id, n.mergedFrom AS mf").data()
    merged_from = {}
    for r in rows:
        mf = r["mf"]
        if not mf:
            merged_from[r["id"]] = []
            continue
        try:
            merged_from[r["id"]] = ast.literal_eval(mf)
        except Exception:
            merged_from[r["id"]] = []
    return merged_from


def main():
    parser = argparse.ArgumentParser(
        description="Check MATCH/NO_MATCH annotations against the live Neo4j graph."
    )
    parser.add_argument("--pairs", required=True,
                        help="Annotated pairs CSV (entity1_type, entity_label_1, "
                             "entity2_type, entity_label_2, label)")
    parser.add_argument("--raw_entities", required=True,
                        help="Pre-fusion entities_final.csv -- only place original "
                             "per-mention text still exists, used for the mergedFrom fallback")
    parser.add_argument("--model", required=True,
                        help="Prefix used to look up {PREFIX}_NEO4J_* in .env, e.g. GPT_V5")
    parser.add_argument("--output", default=None, help="Optional CSV of per-pair verdicts")
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    raw_label_index = load_raw_label_index(args.raw_entities)
    driver = get_driver(args.model)

    direct_cache = {}

    def direct_node_ids(session, etype, label):
        key = TYPE_LABEL_KEY.get(etype)
        if not key or pd.isna(label):
            return set()
        # Cache key mirrors the Cypher match itself (trim + lowercase only) --
        # must NOT share norm()'s quote-stripping, or two distinctly-quoted
        # labels collapse onto one cache slot and whichever is queried first
        # silently overwrites the other's (different) live-match result.
        cache_key = (etype, str(label).strip().lower())
        if cache_key in direct_cache:
            return direct_cache[cache_key]
        query = (
            f"MATCH (n:`{etype}`) "
            f"WHERE toLower(trim(n.`{key}`)) = toLower(trim($label)) "
            f"RETURN n.id AS id"
        )
        with session.begin_transaction() as tx:
            ids = {r["id"] for r in tx.run(query, label=str(label))}
        direct_cache[cache_key] = ids
        return ids

    with driver.session() as session:
        merged_from_map = load_live_merged_from_map(session)

    # raw pre-fusion id -> live final node id, built from the live mergedFrom map
    raw_to_final = {}
    for final_id, raw_ids in merged_from_map.items():
        raw_to_final[final_id] = final_id
        for rid in raw_ids:
            raw_to_final[str(rid).strip()] = final_id

    def merged_from_node_ids(etype, label):
        if pd.isna(label):
            return set()
        raw_ids = raw_label_index.get((etype, norm(label)), [])
        return {raw_to_final[rid] for rid in raw_ids if rid in raw_to_final}

    type_nodes_cache = {}

    def get_type_nodes(session, etype):
        if etype not in type_nodes_cache:
            key = TYPE_LABEL_KEY[etype]
            rows = session.run(
                f"MATCH (n:`{etype}`) RETURN n.id AS id, n.`{key}` AS label"
            ).data()
            type_nodes_cache[etype] = [(r["id"], r["label"]) for r in rows if r["label"]]
        return type_nodes_cache[etype]

    def fuzzy_node_ids(session, etype, label):
        if etype not in FUZZY_TYPES or pd.isna(label):
            return set()
        return {
            nid for nid, nlabel in get_type_nodes(session, etype)
            if token_containment_match(etype, label, nlabel)
        }

    results = []
    with driver.session() as session:
        for _, row in pairs.iterrows():
            label = str(row["label"]).strip().upper()
            t1, t2 = type_col(row, 1), type_col(row, 2)
            l1, l2 = row["entity_label_1"], row["entity_label_2"]

            ids1 = (direct_node_ids(session, t1, l1) or merged_from_node_ids(t1, l1)
                    or fuzzy_node_ids(session, t1, l1))
            ids2 = (direct_node_ids(session, t2, l2) or merged_from_node_ids(t2, l2)
                    or fuzzy_node_ids(session, t2, l2))

            if not ids1 or not ids2:
                verdict = "UNRESOLVED"
            elif label not in ("MATCH", "NO_MATCH"):
                verdict = "SKIPPED_LABEL"
            else:
                same_node = bool(ids1 & ids2)
                correct = same_node if label == "MATCH" else not same_node
                verdict = "CORRECT" if correct else "INCORRECT"

            results.append({
                "entity_type_1": t1,
                "entity_label_1": l1,
                "node_ids_1": ";".join(sorted(ids1)),
                "entity_type_2": t2,
                "entity_label_2": l2,
                "node_ids_2": ";".join(sorted(ids2)),
                "label": label,
                "verdict": verdict,
            })

    driver.close()

    out_df = pd.DataFrame(results)
    print(out_df["verdict"].value_counts())

    if args.output:
        out_df.to_csv(args.output, index=False)
        print(f"Saved per-pair verdicts to {args.output}")

    incorrect = out_df[out_df["verdict"] == "INCORRECT"]
    if len(incorrect):
        print(f"\n{len(incorrect)} INCORRECT case(s):")
        print(incorrect.to_string(index=False))

    unresolved = out_df[out_df["verdict"] == "UNRESOLVED"]
    if len(unresolved):
        print(f"\n{len(unresolved)} UNRESOLVED case(s) (label text matched no live node -- "
              f"can't judge correctness):")
        print(unresolved[["entity_type_1", "entity_label_1", "entity_type_2", "entity_label_2"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
