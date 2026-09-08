# -*- coding: utf-8 -*-
"""
Compute inter-annotator agreement (Cohen's kappa + raw agreement) between the
original annotator's labels (KEY file, held out from the second annotator)
and a second annotator's blind judgments on the same sampled items.

Works for both IAA samples produced for this paper:
  - triples:      status (Correct/Rejected/Edited), and, restricted to items
                   where BOTH annotators rejected, the rejection reason.
  - entity_pairs:  MATCH/NO_MATCH/SKIP label.

Usage:
    # Triple extraction IAA (status agreement + conditional reason agreement)
    python compute_iaa.py --task triples \
        --key data/IAA_samples/triple_extraction_IAA_sample_KEY.csv \
        --blind data/IAA_samples/triple_extraction_IAA_sample_BLIND_completed.csv \
        --annotator_status_col annotator2_status \
        --annotator_reason_col annotator2_reason

    # Entity-pair IAA (GPT-5.1 or Qwen32B -- same script, different files)
    python compute_iaa.py --task entity_pairs \
        --key data/IAA_samples/gpt51_entity_pair_IAA_sample_KEY.csv \
        --blind data/IAA_samples/gpt51_entity_pair_IAA_sample_BLIND_completed.csv \
        --annotator_label_col annotator2_label
"""

import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

KAPPA_BANDS = [
    (0.00, "slight/poor"),
    (0.20, "fair"),
    (0.40, "moderate"),
    (0.60, "substantial"),
    (0.80, "almost perfect"),
]


def interpret_kappa(k):
    band = KAPPA_BANDS[0][1]
    for threshold, label in KAPPA_BANDS:
        if k >= threshold:
            band = label
    return band


def report(name, gold_labels, other_labels):
    n = len(gold_labels)
    if n == 0:
        print(f"\n[{name}] n=0 -- nothing to score, skipping.")
        return
    raw_agree = sum(1 for a, b in zip(gold_labels, other_labels) if a == b) / n
    kappa = cohen_kappa_score(gold_labels, other_labels)
    labels_sorted = sorted(set(gold_labels) | set(other_labels))
    cm = confusion_matrix(gold_labels, other_labels, labels=labels_sorted)

    print(f"\n[{name}] n={n}")
    print(f"  Raw agreement: {raw_agree:.3f} ({raw_agree*100:.1f}%)")
    print(f"  Cohen's kappa: {kappa:.3f} ({interpret_kappa(kappa)})")
    print(f"  Labels: {labels_sorted}")
    print(f"  Confusion matrix (rows=original annotator, cols=second annotator):")
    header = "        " + "".join(f"{l[:10]:>12}" for l in labels_sorted)
    print(header)
    for lbl, row in zip(labels_sorted, cm):
        print(f"  {lbl[:10]:>6}" + "".join(f"{v:>12}" for v in row))


def main():
    p = argparse.ArgumentParser(description="Compute IAA (Cohen's kappa) between two annotators on a sampled item set.")
    p.add_argument("--task", required=True, choices=["triples", "entity_pairs"])
    p.add_argument("--key", required=True, help="KEY csv with item_id + original annotator's label(s)")
    p.add_argument("--blind", required=True, help="Completed BLIND csv with item_id + second annotator's label(s)")
    p.add_argument("--annotator_status_col", default="annotator2_status",
                    help="[triples] column in --blind holding the second annotator's status judgment")
    p.add_argument("--annotator_reason_col", default="annotator2_reason",
                    help="[triples] column in --blind holding the second annotator's reason judgment (when Rejected)")
    p.add_argument("--annotator_label_col", default="annotator2_label",
                    help="[entity_pairs] column in --blind holding the second annotator's MATCH/NO_MATCH/SKIP judgment")
    args = p.parse_args()

    key_df = pd.read_csv(args.key)
    blind_df = pd.read_csv(args.blind)

    if "item_id" not in key_df.columns or "item_id" not in blind_df.columns:
        raise ValueError("Both --key and --blind must have an 'item_id' column to join on.")
    if blind_df["item_id"].duplicated().any():
        raise ValueError("--blind has duplicate item_id values; refusing to guess which row is authoritative.")

    # Pull the annotator's judgment column(s) straight out of --blind BY NAME
    # before touching --key at all, and join by explicit item_id -> value
    # mapping rather than a key/blind merge. This sidesteps a real bug we hit
    # in practice: --key and --blind can easily share column names (e.g. both
    # have "status"/"reason", since that's what the annotation tools write),
    # and a naive merge's auto-suffixing silently makes the unsuffixed name
    # resolve to --key's own column -- so a mistyped --annotator_status_col
    # ends up comparing --key against itself and reports perfect agreement.
    def pull(col):
        if col not in blind_df.columns:
            raise ValueError(f"--blind is missing column '{col}'. Available columns: {blind_df.columns.tolist()}")
        return blind_df.set_index("item_id")[col]

    missing_ids = set(key_df["item_id"]) - set(blind_df["item_id"])
    if missing_ids:
        print(f"WARNING: {len(missing_ids)} item_id(s) in --key were not found in --blind "
              f"(annotator may have skipped rows). Scoring on the {len(key_df) - len(missing_ids)} that matched.")
    scored_key = key_df[key_df["item_id"].isin(blind_df["item_id"])].copy()

    if args.task == "triples":
        other_status_map = pull(args.annotator_status_col)
        gold_status = scored_key["status"].astype(str).str.strip()
        other_status = scored_key["item_id"].map(other_status_map).astype(str).str.strip()
        report("Triple status (Correct/Rejected/Edited)", gold_status.tolist(), other_status.tolist())

        # Reason agreement only makes sense where BOTH annotators judged the
        # triple as something other than Correct -- restricting to their
        # intersection (rather than the original annotator's Rejected set
        # alone) avoids scoring "reason" on items the second annotator
        # thought were fine to begin with.
        if args.annotator_reason_col in blind_df.columns:
            other_reason_map = pull(args.annotator_reason_col)
            both_rejected_mask = (gold_status != "Correct").values & (other_status != "Correct").values
            gold_reason = scored_key.loc[both_rejected_mask, "reason"].fillna("").astype(str).str.strip().str.lower()
            other_reason = scored_key.loc[both_rejected_mask, "item_id"].map(other_reason_map).fillna("").astype(str).str.strip().str.lower()
            report("Rejection reason (restricted to items both annotators rejected)",
                   gold_reason.tolist(), other_reason.tolist())
        else:
            print(f"\n(No '{args.annotator_reason_col}' column found -- skipping reason-agreement scoring.)")

    else:  # entity_pairs
        other_label_map = pull(args.annotator_label_col)
        gold_label = scored_key["label"].astype(str).str.strip().str.upper()
        other_label = scored_key["item_id"].map(other_label_map).astype(str).str.strip().str.upper()
        report("Entity-pair label (MATCH/NO_MATCH/SKIP)", gold_label.tolist(), other_label.tolist())


if __name__ == "__main__":
    main()
