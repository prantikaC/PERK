# -*- coding: utf-8 -*-
"""
Check annotated MATCH/NO_MATCH entity pairs against what actually happened
in the resolved graph:
  - NO_MATCH pair -> the two entities must still be separate final nodes.
  - MATCH pair    -> the two entities must have collapsed into the same
                      final node (not still separate).

Entities in the pairs CSV are identified by (type, label text) -- they come
from a different id space (the golden-triples annotation set) than the
pipeline's own raw extraction ids, so this traces each label back to the
raw pre-fusion entity id(s) sharing that (type, label), then forward to
whichever final node absorbed each raw id via node_fusion.py's mergedFrom
property. A pair is judged by whether entity1's and entity2's resulting
final-node-id sets intersect.

Usage:
    python check_resolution_correctness.py \
        --pairs balanced_entity_all_pairs_50.csv \
        --raw_entities openai_v2_entities_final.csv \
        --fused_entities openai_v2_fused_entities_final.csv \
        --output correctness_report.csv
"""

import argparse
import json
from collections import defaultdict

import pandas as pd

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


def norm(s):
    return str(s).strip().lower()


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


def load_raw_to_final_map(fused_entities_path):
    """raw pre-fusion entity id -> final surviving node id, via mergedFrom."""
    df = pd.read_csv(fused_entities_path)
    raw_to_final = {}
    for _, row in df.iterrows():
        final_id = str(row["id"]).strip()
        raw_to_final[final_id] = final_id
        try:
            props = json.loads(row["properties"])
        except Exception:
            props = {}
        for merged_id in props.get("mergedFrom") or []:
            raw_to_final[str(merged_id).strip()] = final_id
    return raw_to_final


def resolve_final_ids(etype, label, raw_label_index, raw_to_final):
    raw_ids = raw_label_index.get((etype, norm(label)), [])
    return {raw_to_final[rid] for rid in raw_ids if rid in raw_to_final}


def main():
    parser = argparse.ArgumentParser(
        description="Check MATCH/NO_MATCH annotations against the actual resolved graph."
    )
    parser.add_argument("--pairs", required=True,
                        help="Annotated pairs CSV (entity1_type, entity_label_1, "
                             "entity2_type, entity_label_2, label)")
    parser.add_argument("--raw_entities", required=True,
                        help="Pre-fusion entities_final.csv (raw extraction ids)")
    parser.add_argument("--fused_entities", required=True,
                        help="Post-fusion fused_entities_final.csv (has mergedFrom)")
    parser.add_argument("--output", default=None, help="Optional CSV of per-pair verdicts")
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    raw_label_index = load_raw_label_index(args.raw_entities)
    raw_to_final = load_raw_to_final_map(args.fused_entities)

    def type_col(row, n):
        for candidate in (f"entity{n}_type", "entity_type"):
            if candidate in row and pd.notna(row[candidate]):
                return row[candidate]
        return None

    results = []
    for _, row in pairs.iterrows():
        label = str(row["label"]).strip().upper()
        t1, t2 = type_col(row, 1), type_col(row, 2)
        l1, l2 = row["entity_label_1"], row["entity_label_2"]

        final1 = resolve_final_ids(t1, l1, raw_label_index, raw_to_final)
        final2 = resolve_final_ids(t2, l2, raw_label_index, raw_to_final)

        if not final1 or not final2:
            verdict = "UNRESOLVED"
        elif label not in ("MATCH", "NO_MATCH"):
            verdict = "SKIPPED_LABEL"
        else:
            same_node = bool(final1 & final2)
            correct = same_node if label == "MATCH" else not same_node
            verdict = "CORRECT" if correct else "INCORRECT"

        results.append({
            "entity_type_1": t1,
            "entity_label_1": l1,
            "final_ids_1": ";".join(sorted(final1)),
            "entity_type_2": t2,
            "entity_label_2": l2,
            "final_ids_2": ";".join(sorted(final2)),
            "label": label,
            "verdict": verdict,
        })

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
        print(f"\n{len(unresolved)} UNRESOLVED case(s) (label text not found in raw extraction -- "
              f"can't judge correctness):")
        print(unresolved[["entity_type_1", "entity_label_1", "entity_type_2", "entity_label_2"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
