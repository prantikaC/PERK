# -*- coding: utf-8 -*-
import argparse
import logging

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


def main():
    parser = argparse.ArgumentParser(description="End-to-End Entity Resolution Pipeline Evaluation")
    parser.add_argument("--golden_set",    required=True, help="Annotated golden set CSV")
    parser.add_argument("--llm_resolved",  required=True, help="LLM resolved output CSV")
    parser.add_argument("--grey_zone",     required=True, help="Original grey zone CSV (for FAISS rank calculation)")
    parser.add_argument("--errors_output", required=True, help="Output CSV for misclassified pairs")
    parser.add_argument("--log",           default="pipeline_evaluation.log")
    args = parser.parse_args()

    setup_logger(args.log)
    logging.info("Starting End-to-End Pipeline Evaluation...")

    # Load and filter golden set
    df_golden = pd.read_csv(args.golden_set)
    df_golden['label'] = df_golden['label'].astype(str).str.strip().str.upper()
    df_golden = df_golden[df_golden['label'].isin(['MATCH', 'NO_MATCH'])].copy()

    df_llm = pd.read_csv(args.llm_resolved)

    merge_cols = ['entity_label_1', 'evidence_1', 'entity_label_2', 'evidence_2']
    for col in merge_cols:
        df_golden[col] = df_golden[col].astype(str).str.strip()
        df_llm[col]    = df_llm[col].astype(str).str.strip()

    df_eval = pd.merge(df_golden, df_llm[merge_cols + ['llm_prediction']], on=merge_cols, how='inner')
    if len(df_eval) == 0:
        logging.error("Could not match any rows between golden set and LLM output. Check column names.")
        return

    # LLM metrics
    y_true = (df_eval['label'] == 'MATCH').astype(int)
    y_pred = (df_eval['llm_prediction'] == 'MATCH').astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    # FAISS Recall@K — compute true ranks from original grey zone
    logging.info("Calculating true FAISS ranks from grey zone file...")
    df_grey = pd.read_csv(args.grey_zone)
    for col in merge_cols:
        df_grey[col] = df_grey[col].astype(str).str.strip()

    df_grey = df_grey.sort_values(
        by=['entity_label_1', 'similarity_score'], ascending=[True, False]
    )
    df_grey['rank'] = df_grey.groupby('entity_label_1').cumcount() + 1
    df_ranks = df_grey[merge_cols + ['rank']].drop_duplicates(subset=merge_cols)
    df_eval  = pd.merge(df_eval, df_ranks, on=merge_cols, how='left')

    true_matches        = df_eval[df_eval['label'] == 'MATCH']
    total_true_matches  = len(true_matches)

    # Error analysis
    df_errors = df_eval[df_eval['label'] != df_eval['llm_prediction']].copy()
    df_errors.to_csv(args.errors_output, index=False)

    # Report
    logging.info("=" * 50)
    logging.info(" END-TO-END PIPELINE PERFORMANCE")
    logging.info("=" * 50)
    logging.info(f"Precision : {precision:.4f}")
    logging.info(f"Recall    : {recall:.4f}")
    logging.info(f"F1-Score  : {f1:.4f}")
    logging.info("-" * 50)

    if total_true_matches > 0:
        logging.info("RECALL @ K (FAISS Neighbourhood)")
        for k in range(1, 11):
            matches_within_k = len(true_matches[true_matches['rank'] <= k])
            recall_at_k = matches_within_k / total_true_matches
            logging.info(f"  Recall @ {k:2d}: {recall_at_k * 100:.1f}%")
            if recall_at_k == 1.0:
                logging.info(f"  -> 100% of true matches captured at K={k}.")
                break
    else:
        logging.info("No true matches found to calculate Recall@K.")

    logging.info("-" * 50)
    logging.info(f"Misclassified pairs: {len(df_errors)} saved to {args.errors_output}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
