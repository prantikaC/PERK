# -*- coding: utf-8 -*-
"""
Generate per-metric F1 comparison plots (Evidence vs Full-Email context)
across all extraction models. Saves one PDF per metric.

Usage:
    python plot_results.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

models = ["GPT-5.1", "Llama\n3.1-8B", "Gemma\n3-4B", "Qwen2.5\n-7B", "Qwen2.5\n-32B"]
x = np.arange(len(models))

data = {
    "Sub": {
        "Evidence":   [0.79, 0.31, 0.29, 0.31, 0.47],
        "Full-Email": [0.88, 0.84, 0.86, 0.84, 0.83],
    },
    "Obj": {
        "Evidence":   [0.83, 0.22, 0.18, 0.35, 0.38],
        "Full-Email": [0.84, 0.60, 0.60, 0.66, 0.69],
    },
    "Ent": {
        "Evidence":   [0.82, 0.27, 0.25, 0.34, 0.45],
        "Full-Email": [0.87, 0.74, 0.78, 0.77, 0.80],
    },
    "Pred": {
        "Evidence":   [0.84, 0.33, 0.37, 0.49, 0.55],
        "Full-Email": [0.84, 0.33, 0.37, 0.49, 0.55],
    },
    "Tri": {
        "Evidence":   [0.80, 0.09, 0.07, 0.16, 0.28],
        "Full-Email": [0.81, 0.14, 0.11, 0.21, 0.33],
    },
}

titles = {
    "Sub":  "Subject F1",
    "Obj":  "Object F1",
    "Ent":  "Entity F1",
    "Pred": "Relation F1",
    "Tri":  "Triple F1",
}
filenames = {k: f"fig_{k.lower()}.pdf" for k in titles}

BLUE = "#2563EB"
RED  = "#DC2626"
MARKER_KW = dict(markersize=5, linewidth=1.6)

for key, title in titles.items():
    fig, ax = plt.subplots(figsize=(3.2, 2.5))

    ev = data[key]["Evidence"]
    fe = data[key]["Full-Email"]

    ls_ev = "--" if ev == fe else "-"

    ax.plot(x, fe, color=RED,  marker="s", linestyle="-",   label="Full-Email", **MARKER_KW)
    ax.plot(x, ev, color=BLUE, marker="o", linestyle=ls_ev, label="Evidence",   **MARKER_KW)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7)
    ax.set_ylabel("F1 Score", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    if key == "Sub":
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    fig.tight_layout(pad=0.4)
    fig.savefig(filenames[key], dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filenames[key]}")

print("Done.")
