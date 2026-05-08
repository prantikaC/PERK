# -*- coding: utf-8 -*-
"""
Evaluate extracted KG triples against a manually annotated golden set.
Computes per-email and micro-averaged Precision, Recall, F1.

Usage:
    python evaluate_triples.py \
        --golden     triples_annotated_GOLDEN.csv \
        --system_dir /path/to/system/output/ \
        --threshold  0.85
"""

import argparse
import os
from difflib import SequenceMatcher

import pandas as pd


def similar(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def is_match(t1, t2, threshold):
    if t1["predicate"].strip().lower() != t2["predicate"].strip().lower():
        return False
    return similar(t1["subject"], t2["subject"]) >= threshold \
       and similar(t1["object"],  t2["object"])  >= threshold


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate extracted triples against a golden annotated set."
    )
    parser.add_argument("--golden",     required=True,
                        help="Golden CSV with 'original_source_id' column")
    parser.add_argument("--system_dir", required=True,
                        help="Directory containing triples_emailN.csv files")
    parser.add_argument("--threshold",  type=float, default=0.85,
                        help="Fuzzy match threshold for subject/object (default: 0.85)")
    args = parser.parse_args()

    print(f"Loading golden truth from: {args.golden}")
    df_gold = pd.read_csv(args.golden)

    if "original_source_id" not in df_gold.columns:
        raise ValueError(
            "Golden CSV missing 'original_source_id' column. "
            "Ensure the annotation tool exported this column."
        )

    verified_ids = sorted(df_gold["original_source_id"].unique().tolist())
    print(f"Found golden truth for {len(verified_ids)} emails.")
    print(f"Evaluating system: {os.path.basename(args.system_dir)}")
    print("-" * 80)
    print(f"{'Email':<10} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'TP':<4} {'FP':<4} {'FN':<4}")
    print("-" * 80)

    total_tp = total_fp = total_fn = 0

    for email_id in verified_ids:
        gold_subset = df_gold[df_gold["original_source_id"] == email_id].to_dict("records")

        try:
            num      = email_id.replace("e", "")
            sys_file = os.path.join(args.system_dir, f"triples_email{num}.csv")
            sys_subset = (
                pd.read_csv(sys_file).to_dict("records")
                if os.path.exists(sys_file)
                else []
            )
        except Exception as e:
            print(f"Error loading {email_id}: {e}")
            sys_subset = []

        matched_sys = set()
        tp = fn = 0

        for g in gold_subset:
            found = False
            for idx, s in enumerate(sys_subset):
                if idx in matched_sys:
                    continue
                if is_match(g, s, args.threshold):
                    found = True
                    matched_sys.add(idx)
                    break
            if found:
                tp += 1
            else:
                fn += 1

        fp = len(sys_subset) - len(matched_sys)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"{email_id:<10} | {prec:.2f}     | {rec:.2f}     | {f1:.2f}     | {tp:<4} {fp:<4} {fn:<4}")

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1   = (
        2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) > 0 else 0
    )

    print("=" * 80)
    print(f"EVALUATION RESULTS FOR: {os.path.basename(args.system_dir)}")
    print(f"Verified Emails: {len(verified_ids)}")
    print("-" * 30)
    print(f"MICRO PRECISION: {micro_prec:.4f}")
    print(f"MICRO RECALL:    {micro_rec:.4f}")
    print(f"MICRO F1 SCORE:  {micro_f1:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
