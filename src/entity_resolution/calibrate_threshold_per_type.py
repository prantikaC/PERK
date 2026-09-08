# -*- coding: utf-8 -*-
"""
Calibrate a separate FAISS auto-reject threshold per entity_type, instead of
one global number for every type. calibrate_threshold.py's single 0.6547
floor was derived across all types pooled together; that generalizes fine
for types with few golden examples, but Task massively dominates faiss
blocking's grey-zone volume (33,943 / 38,530 candidates on the openai_v2
corpus) with many short, generic, topically-similar-but-distinct phrases
scoring well above 0.6547 under a general-purpose sentence embedding. A
type-specific floor lets Task demand a higher bar without moving the
threshold for types the global number already works for.

Requires >= MIN_MATCH_EXAMPLES golden MATCH pairs for a type before trusting
its own calibrated threshold; types below that (or entirely absent from the
golden set) fall back to the global auto-reject threshold instead of a
number computed from too few positives to be stable.

Usage:
    python calibrate_threshold_per_type.py \
        --golden_set openai_golden_entity_pair.csv \
        --global_threshold 0.6547 \
        --log_output per_type_threshold_calibration_log.txt
"""

import argparse
import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

MIN_MATCH_EXAMPLES = 5


def auto_reject_threshold(scores, y_true):
    thresholds = np.linspace(0.0, 1.0, 1000)
    recalls = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
    valid = [thresholds[i] for i in range(len(thresholds)) if recalls[i] >= 0.99]
    return max(valid) if valid else None


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate a per-entity_type FAISS auto-reject threshold from golden pairs."
    )
    parser.add_argument("--golden_set", required=True)
    parser.add_argument("--global_threshold", type=float, required=True,
                        help="Fallback threshold for types with too few golden MATCH examples "
                             f"(< {MIN_MATCH_EXAMPLES}) to calibrate on their own.")
    parser.add_argument("--log_output", default="per_type_threshold_calibration_log.txt")
    parser.add_argument("--output", default="per_type_thresholds.csv",
                         help="CSV of entity_type,threshold,source -- consumed by "
                              "faiss_blocking.py's --threshold_map")
    args = parser.parse_args()

    df = pd.read_csv(args.golden_set)
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    df = df[df['label'].isin(['MATCH', 'NO_MATCH'])].copy()
    df['y_true'] = (df['label'] == 'MATCH').astype(int)

    rows = []
    with open(args.log_output, "w") as log_fh:
        log_fh.write(f"Calibration run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_fh.write(f"Global fallback threshold: {args.global_threshold}\n")
        log_fh.write(f"Minimum golden MATCH examples required to trust a per-type threshold: "
                     f"{MIN_MATCH_EXAMPLES}\n\n")

        for entity_type, df_t in df.groupby('entity_type'):
            n_match = int(df_t['y_true'].sum())
            n_total = len(df_t)
            log_fh.write(f"{'='*50}\nENTITY TYPE: {entity_type}\n{'='*50}\n")
            log_fh.write(f"Golden pairs: {n_total} ({n_match} MATCH, {n_total - n_match} NO_MATCH)\n")

            if n_match < MIN_MATCH_EXAMPLES:
                log_fh.write(f"Too few MATCH examples (< {MIN_MATCH_EXAMPLES}) to calibrate "
                             f"a stable threshold -- falling back to global {args.global_threshold}.\n\n")
                rows.append({"entity_type": entity_type, "threshold": args.global_threshold,
                             "source": f"fallback (only {n_match} golden MATCH examples)"})
                continue

            t = auto_reject_threshold(df_t['similarity_score'].values, df_t['y_true'].values)
            if t is None:
                log_fh.write(f"No threshold achieves 99% recall -- falling back to global "
                             f"{args.global_threshold}.\n\n")
                rows.append({"entity_type": entity_type, "threshold": args.global_threshold,
                             "source": "fallback (no valid 99%-recall threshold)"})
                continue

            log_fh.write(f"AUTO-REJECT THRESHOLD: {t:.4f} (>=99% recall, calibrated on this "
                         f"type's own {n_match} golden MATCH examples)\n\n")
            rows.append({"entity_type": entity_type, "threshold": round(float(t), 4),
                         "source": f"calibrated ({n_match} golden MATCH examples)"})

    present_types = set(df['entity_type'].unique())
    log_fh_note = (f"\nTypes not present in the golden set at all (Person, Team, Organization, "
                    f"Conference, Journal, ...) are not listed here -- faiss_blocking.py falls "
                    f"back to --threshold ({args.global_threshold}) for any type missing from "
                    f"this map.\n")
    with open(args.log_output, "a") as log_fh:
        log_fh.write(log_fh_note)

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Log saved to {args.log_output}")
    print(f"Per-type threshold map saved to {args.output}")


if __name__ == "__main__":
    main()
