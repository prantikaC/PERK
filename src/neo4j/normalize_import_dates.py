# -*- coding: utf-8 -*-
"""
Normalize *Date columns in an already-split neo4j_import/ directory (the
:ID/:LABEL per-type CSV format) to ISO YYYY-MM-DD, writing a NEW import dir.

Use this to apply date normalization to a graph that is already in import form
without re-running the full extraction/ER pipeline (option B). Reuses the exact
normalize_date() logic from prepare_import.py, so it matches the pipeline.

Usage:
    python normalize_import_dates.py \
        --in_dir  datasets/neo4j_import \
        --out_dir datasets/neo4j_import_datenorm
"""

import argparse
import csv
import os
import shutil
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_import import normalize_date  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="ISO-normalize *Date columns in a neo4j_import dir.")
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    in_ent = os.path.join(args.in_dir, "entities")
    in_rel = os.path.join(args.in_dir, "relations")
    out_ent = os.path.join(args.out_dir, "entities")
    out_rel = os.path.join(args.out_dir, "relations")
    os.makedirs(out_ent, exist_ok=True)
    os.makedirs(out_rel, exist_ok=True)

    # relations unchanged
    for fn in os.listdir(in_rel):
        if fn.endswith(".csv"):
            shutil.copy2(os.path.join(in_rel, fn), os.path.join(out_rel, fn))

    stats = Counter()
    for fn in sorted(os.listdir(in_ent)):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(in_ent, fn), encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            date_cols = [i for i, h in enumerate(header) if h.lower().endswith("date")]
            out_rows = [header]
            for row in reader:
                for i in date_cols:
                    if i < len(row):
                        raw = str(row[i]).strip()
                        nv = normalize_date(row[i])
                        if raw and not nv:
                            stats[f"{header[i]}:blanked"] += 1
                        elif raw and nv != raw:
                            stats[f"{header[i]}:normalized"] += 1
                        elif raw:
                            stats[f"{header[i]}:kept"] += 1
                        row[i] = nv
                out_rows.append(row)
        with open(os.path.join(out_ent, fn), "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(out_rows)

    print(f"Wrote normalized import to {args.out_dir}")
    if stats:
        print("date normalization:")
        for k in sorted(stats):
            print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    main()
