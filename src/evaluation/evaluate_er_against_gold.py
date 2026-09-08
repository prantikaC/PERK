# -*- coding: utf-8 -*-
"""
Evaluates entity resolution on a fused graph against the 50-entity/1,225-pair
golden ER benchmark (data/entity_resolution/golden_triples_er/).

Methodology: for each gold pair, resolve both sides to a canonical id, then
check gold MATCH pairs land on the same canonical id and gold NO_MATCH pairs
land on different ones. Any pair where either side isn't locatable is
excluded, not imputed.

Locatability walks the FULL mergedFrom provenance chain back to each raw
id's ORIGINAL label (via --raw_entities, the pre-fusion entities file) --
not just the canonical node's current, possibly-collapsed property value.
This matters: when two labels merge (e.g. "JOCCH" and "ACM Journal on
Computing and Cultural Heritage (JOCCH)" collapse into one canonical row),
the canonical's own property is now a SINGLE string -- a gold entity whose
label was the one that DIDN'T survive as the collapsed string would
otherwise wrongly look "not locatable" even though the merge that absorbed
it was exactly correct. Confirmed empirically: this was silently excluding
the one gold Journal MATCH pair specifically because the acronym-merge
round correctly fused it.

Reports full two-class precision/recall/support (MATCH == graph MERGE,
NO_MATCH == graph keeps them as separate nodes), not just the MATCH-class
number -- given gold NO_MATCH pairs vastly outnumber MATCH pairs (1223 vs 2
in the current benchmark), the NO_MATCH-class numbers are equally
informative and were previously not printed at all.

Usage:
    python evaluate_er_against_gold.py --name "GPT-5.1-PERK" \
        --fused_entities data/entity_resolution/openai_v2/openai_v2_fused_entities_final_acronymmerged.csv \
        --raw_entities   data/entity_resolution/openai_v2/openai_v2_entities_final_normdates.csv
"""
import argparse
import json

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

TYPE_LABEL_KEY = {
    'Person': 'personName', 'Team': 'teamName', 'Organization': 'orgName',
    'Paper': 'paperTitle', 'Conference': 'confTitle', 'Journal': 'journalTitle',
    'SubmissionID': 'identifier', 'PaperStatus': 'statusType', 'Meeting': 'meetAgenda',
    'Dataset': 'datasetName', 'Method': 'methodName', 'Task': 'taskName', 'Metric': 'metricName',
}


def norm(s):
    return str(s).strip().lower()


def build_raw_label_lookup(raw_entities_path):
    """raw_id -> {type, label} straight off the PRE-fusion entities file --
    every individual mention's own original label, before any merge ever
    collapsed it into something else."""
    raw = pd.read_csv(raw_entities_path)
    lookup = {}
    for _, row in raw.iterrows():
        etype = row['type']
        key_prop = TYPE_LABEL_KEY.get(etype)
        if not key_prop:
            continue
        try:
            props = json.loads(row['properties'])
        except Exception:
            continue
        label = props.get(key_prop)
        if label is not None:
            lookup[row['id']] = (etype, label)
    return lookup


def build_label_to_canonical(fused_entities_path, raw_label_lookup):
    """(norm(label), type) -> set of canonical ids. For each surviving
    (canonical) row, indexes its own CURRENT label AND, for every raw id in
    its full mergedFrom chain (both this round's and any prior round's,
    already unioned together by apply_acronym_merges.py /
    node_fusion.py), that raw id's ORIGINAL pre-fusion label -- so a gold
    entity is locatable via ANY label it was ever known by, not just
    whatever string happens to survive as the final collapsed property."""
    fused = pd.read_csv(fused_entities_path)
    lookup = {}

    def register(label, etype, canonical_id):
        if label is None:
            return
        labels = label if isinstance(label, list) else [label]
        for l in labels:
            lookup.setdefault((norm(l), etype), set()).add(canonical_id)

    for _, row in fused.iterrows():
        etype = row['type']
        key_prop = TYPE_LABEL_KEY.get(etype)
        canonical_id = row['id']
        try:
            props = json.loads(row['properties'])
        except Exception:
            props = {}

        if key_prop:
            register(props.get(key_prop), etype, canonical_id)

        merged_from = props.get('mergedFrom', [])
        if isinstance(merged_from, list):
            for raw_id in merged_from:
                raw_entry = raw_label_lookup.get(raw_id)
                if raw_entry:
                    raw_type, raw_label = raw_entry
                    register(raw_label, raw_type, canonical_id)

    return lookup


def evaluate_graph(name, fused_entities_path, raw_entities_path, subset50, allpairs):
    raw_label_lookup = build_raw_label_lookup(raw_entities_path)
    label_to_canonical = build_label_to_canonical(fused_entities_path, raw_label_lookup)

    gold_to_canonical = {}
    for _, r in subset50.iterrows():
        gold_to_canonical[r['id']] = label_to_canonical.get((norm(r['label']), r['type']), set())

    print(f"\n=== {name} ===")
    found = sum(1 for v in gold_to_canonical.values() if v)
    print(f"Gold-sample entities locatable in this fused graph: {found} / {len(subset50)}")

    y_true, y_pred = [], []
    skipped = 0
    for _, r in allpairs.iterrows():
        c1 = gold_to_canonical.get(r['entity1_id'], set())
        c2 = gold_to_canonical.get(r['entity2_id'], set())
        if not c1 or not c2:
            skipped += 1
            continue  # exclude, don't impute
        y_true.append(1 if r['label'] == 'MATCH' else 0)
        y_pred.append(1 if (c1 & c2) else 0)

    n_eval = len(y_true)
    print(f"Pairs evaluated (both sides locatable): {n_eval} / {len(allpairs)}  (excluded: {skipped})")
    n_gold_match_eval = sum(y_true)
    print(f"Of the evaluated pairs, gold MATCH: {n_gold_match_eval}  |  gold NO_MATCH: {n_eval - n_gold_match_eval}")
    if not n_eval:
        return

    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    print(f"Accuracy: {correct}/{n_eval} = {correct/n_eval:.4f}")

    print("\nFull two-class report (MATCH == graph MERGE, NO_MATCH == graph keeps separate):")
    print(classification_report(
        y_true, y_pred, labels=[1, 0], target_names=["MATCH/MERGE", "NO_MATCH/NO_MERGE"],
        zero_division=0, digits=4,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    print("Confusion matrix (rows=gold, cols=predicted):")
    print(f"                    pred MERGE   pred NO_MERGE")
    print(f"  gold MATCH        {cm[0][0]:>10}   {cm[0][1]:>13}")
    print(f"  gold NO_MATCH     {cm[1][0]:>10}   {cm[1][1]:>13}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ER on a fused graph against the golden 50-entity/1225-pair benchmark")
    parser.add_argument("--name", required=True, help="Label for this run in the printed report")
    parser.add_argument("--fused_entities", required=True, help="Fused entities CSV to evaluate")
    parser.add_argument("--raw_entities", required=True,
                         help="PRE-fusion entities CSV (e.g. {prefix}_entities_final_normdates.csv) -- "
                              "used to recover each merged-away raw id's original label for locatability.")
    parser.add_argument("--subset50",
                         default="data/entity_resolution/golden_triples_er/faiss_blocked_entity_subset_50.csv",
                         help="Golden 50-entity subset CSV")
    parser.add_argument("--allpairs",
                         default="data/entity_resolution/golden_triples_er/faiss_blocked_entity_all_pairs_50.csv",
                         help="Golden 1225-pair CSV")
    args = parser.parse_args()

    subset50 = pd.read_csv(args.subset50)
    allpairs = pd.read_csv(args.allpairs)
    evaluate_graph(args.name, args.fused_entities, args.raw_entities, subset50, allpairs)


if __name__ == "__main__":
    main()
