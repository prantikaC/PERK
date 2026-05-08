# -*- coding: utf-8 -*-
"""
Calibrate FAISS blocking similarity threshold from annotated entity pairs.
Computes precision-recall curves and finds:
  - Auto-reject threshold: highest score where recall >= 99%
  - Auto-match threshold:  lowest score where precision >= 98%

Usage:
    python calibrate_threshold.py \
        --datasets OpenAI=openai_annotated.csv Qwen=qwen32b_annotated.csv \
        --plot_output calibration_plot.png \
        --log_output calibration_log.txt
"""

import argparse
import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def process_calibration_data(filepath, dataset_name, log_fh):
    df = pd.read_csv(filepath)
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    df_valid = df[df['label'].isin(['MATCH', 'NO_MATCH'])].copy()

    total_pairs = len(df_valid)
    df_valid['y_true'] = (df_valid['label'] == 'MATCH').astype(int)
    scores = df_valid['similarity_score'].values
    y_true = df_valid['y_true'].values

    thresholds = np.linspace(0.0, 1.0, 1000)
    precisions, recalls = [], []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=1))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))

    valid_upper = [thresholds[i] for i in range(len(thresholds))
                   if precisions[i] >= 0.98 and recalls[i] > 0]
    auto_match_t = min(valid_upper) if valid_upper else None

    valid_lower = [thresholds[i] for i in range(len(thresholds)) if recalls[i] >= 0.99]
    auto_reject_t = max(valid_lower) if valid_lower else None

    log_fh.write(f"{'='*50}\nDATASET: {dataset_name}\n{'='*50}\n")
    log_fh.write(f"Total annotated pairs: {total_pairs}\n\n")
    if auto_match_t is not None:
        log_fh.write(f"AUTO-MATCH THRESHOLD:  {auto_match_t:.4f}  "
                     f"(>=98% precision, bypass LLM and auto-accept)\n\n")
    else:
        log_fh.write("AUTO-MATCH: no threshold achieves 98% precision.\n\n")
    if auto_reject_t is not None:
        log_fh.write(f"AUTO-REJECT THRESHOLD: {auto_reject_t:.4f}  "
                     f"(>=99% recall, bypass LLM and auto-reject)\n\n")
    else:
        log_fh.write("AUTO-REJECT: no valid threshold found.\n\n")
    log_fh.write("\n")

    return thresholds, precisions, recalls, auto_match_t, auto_reject_t


def plot_axis(ax, title, thresh, prec, rec, match_t, reject_t):
    ax.plot(thresh, prec, label='Precision', color='#1f77b4', linewidth=2.5)
    ax.plot(thresh, rec,  label='Recall',    color='#ff7f0e', linewidth=2.5)
    if reject_t is not None:
        ax.axvline(x=reject_t, color='red',   linestyle='--', linewidth=1.5,
                   label=f'Auto-Reject ($\\tau={reject_t:.4f}$)')
        ax.axvspan(0, reject_t, color='red', alpha=0.1)
    if match_t is not None:
        ax.axvline(x=match_t, color='green', linestyle='--', linewidth=1.5,
                   label=f'Auto-Match ($\\tau={match_t:.4f}$)')
        ax.axvspan(match_t, 1, color='green', alpha=0.1)
    grey_end = match_t if match_t is not None else 1.0
    if reject_t is not None:
        ax.axvspan(reject_t, grey_end, color='gray', alpha=0.15, label='LLM Grey Zone')
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('Cosine Similarity Score', fontsize=12)
    ax.set_ylabel('Metric Score', fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9, edgecolor='black')


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate FAISS blocking threshold from annotated entity pairs."
    )
    parser.add_argument("--datasets", nargs='+', required=True,
                        metavar="NAME=FILE",
                        help="One or two datasets as Name=path pairs, "
                             "e.g. OpenAI=openai_annotated.csv Qwen=qwen32b_annotated.csv")
    parser.add_argument("--plot_output", default="calibration_plot.png")
    parser.add_argument("--log_output",  default="calibration_log.txt")
    args = parser.parse_args()

    datasets = {}
    for item in args.datasets:
        name, path = item.split("=", 1)
        datasets[name] = path

    if len(datasets) > 2:
        parser.error("At most 2 datasets supported for side-by-side plotting.")

    results = {}
    with open(args.log_output, "w") as log_fh:
        log_fh.write(f"Calibration run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for name, path in datasets.items():
            print(f"Processing {name} ({path})...")
            results[name] = process_calibration_data(path, name, log_fh)
    print(f"Log saved to {args.log_output}")

    names = list(results.keys())
    ncols = len(names)
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6), dpi=300)
    if ncols == 1:
        axes = [axes]

    labels = ['(a)', '(b)', '(c)']
    for ax, label, name in zip(axes, labels, names):
        thresh, prec, rec, match_t, reject_t = results[name]
        plot_axis(ax, f"{label} {name} Pipeline", thresh, prec, rec, match_t, reject_t)

    plt.tight_layout()
    plt.savefig(args.plot_output, format='png', bbox_inches='tight')
    plt.savefig(args.plot_output.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Plots saved to {args.plot_output} (and .pdf)")


if __name__ == "__main__":
    main()
