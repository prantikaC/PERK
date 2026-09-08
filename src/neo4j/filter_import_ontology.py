# -*- coding: utf-8 -*-
"""
Filter a prepared neo4j_import/ directory so that only ontology-conforming
relations survive, writing a NEW import dir (v1 is left untouched).

Applies the same rule as clean_kg.py's is_valid_triplet: a relation row is kept
only if (start_type, end_type) is a declared valid_pair for its relation type
(forward direction only). Entities are copied through unchanged.

Usage:
    python filter_import_ontology.py \
        --in_dir   datasets/neo4j_import \
        --out_dir  datasets/neo4j_import_v2 \
        --ontology ontology/PERKOnto.json
"""

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Filter neo4j_import relations by ontology valid_pairs.")
    p.add_argument("--in_dir", required=True, help="Existing neo4j_import dir (entities/ + relations/)")
    p.add_argument("--out_dir", required=True, help="New import dir to write (must not be in_dir)")
    p.add_argument("--ontology", default="PERKOnto.json")
    return p.parse_args()


def load_valid_pairs(path):
    onto = json.load(open(path, encoding="utf-8"))
    valid = defaultdict(set)
    for rel_name, rel in onto.get("relationships", {}).items():
        rtype = rel.get("type", rel_name)
        for pair in rel.get("valid_pairs", []):
            if len(pair) == 2:
                valid[rtype].add((pair[0], pair[1]))
    return valid


def build_id_to_type(entities_dir):
    """Map node :ID -> :LABEL across every entity CSV."""
    id_to_type = {}
    for fn in os.listdir(entities_dir):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(entities_dir, fn), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                id_to_type[row[":ID"]] = row[":LABEL"]
    return id_to_type


def main():
    args = parse_args()
    in_ent = os.path.join(args.in_dir, "entities")
    in_rel = os.path.join(args.in_dir, "relations")
    out_ent = os.path.join(args.out_dir, "entities")
    out_rel = os.path.join(args.out_dir, "relations")
    os.makedirs(out_ent, exist_ok=True)
    os.makedirs(out_rel, exist_ok=True)

    valid = load_valid_pairs(args.ontology)
    id_to_type = build_id_to_type(in_ent)
    print(f"Loaded {len(id_to_type)} entities, {len(valid)} relation types with valid_pairs")

    # entities: copy through unchanged
    for fn in os.listdir(in_ent):
        if fn.endswith(".csv"):
            shutil.copy2(os.path.join(in_ent, fn), os.path.join(out_ent, fn))

    total_in = total_out = 0
    print("\nRelation                 kept /  in   dropped")
    print("-" * 48)
    for fn in sorted(os.listdir(in_rel)):
        if not fn.endswith(".csv"):
            continue
        rtype = fn[:-4]
        kept = 0
        n_in = 0
        with open(os.path.join(in_rel, fn), encoding="utf-8") as f_in, \
             open(os.path.join(out_rel, fn), "w", encoding="utf-8", newline="") as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                n_in += 1
                s = id_to_type.get(row[":START_ID"])
                o = id_to_type.get(row[":END_ID"])
                if s is not None and o is not None and (s, o) in valid.get(rtype, set()):
                    writer.writerow(row)
                    kept += 1
        total_in += n_in
        total_out += kept
        print(f"{rtype:22s} {kept:6d} / {n_in:5d}  {n_in - kept:6d}")

    print("-" * 48)
    print(f"{'TOTAL':22s} {total_out:6d} / {total_in:5d}  {total_in - total_out:6d}")
    print(f"\nWrote ontology-conforming import to {args.out_dir}")


if __name__ == "__main__":
    main()
