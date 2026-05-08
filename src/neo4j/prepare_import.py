# -*- coding: utf-8 -*-
"""
Convert master entities/relations CSVs into per-type CSVs
in Neo4j bulk import format.

Output structure:
    <output>/
    ├── entities/
    │   ├── Person.csv
    │   ├── Paper.csv
    │   └── ...
    └── relations/
        ├── sentBy.csv
        ├── hasAuthor.csv
        └── ...

Usage:
    python prepare_import.py \
        --entities  entities_clean.csv \
        --relations relations_clean.csv \
        --output    neo4j_import/
"""

import argparse
import ast
import csv
import os


def convert_entities(entities_file, output_dir):
    with open(entities_file, "r", encoding="utf-8") as f:
        entities = list(csv.DictReader(f))

    grouped = {}
    for row in entities:
        etype = row["type"]
        try:
            props = ast.literal_eval(row["properties"])
        except Exception:
            props = {}
        grouped.setdefault(etype, []).append((row["id"], props))

    for etype, rows in grouped.items():
        all_keys = sorted({k for _, props in rows for k in props})
        filepath = os.path.join(output_dir, f"{etype}.csv")
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([":ID"] + all_keys + [":LABEL"])
            for eid, props in rows:
                writer.writerow([eid] + [props.get(k, "") for k in all_keys] + [etype])

    print(f"  {len(grouped)} entity type file(s) written to {output_dir}/")


def convert_relations(relations_file, output_dir):
    with open(relations_file, "r", encoding="utf-8") as f:
        relations = list(csv.DictReader(f))

    grouped = {}
    for row in relations:
        rel = row["relation"]
        grouped.setdefault(rel, []).append((
            row["start_id"], row["end_id"],
            row.get("context", ""), row.get("source", "")
        ))

    for rel, rows in grouped.items():
        filepath = os.path.join(output_dir, f"{rel}.csv")
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([":START_ID", ":END_ID", ":TYPE", "context", "source"])
            for start_id, end_id, context, source in rows:
                writer.writerow([start_id, end_id, rel, context, source])

    print(f"  {len(grouped)} relation type file(s) written to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Convert entities/relations CSVs to Neo4j bulk import format."
    )
    parser.add_argument("--entities",  required=True, help="Cleaned entities CSV")
    parser.add_argument("--relations", required=True, help="Cleaned relations CSV")
    parser.add_argument("--output",    required=True, help="Base output directory")
    args = parser.parse_args()

    entities_dir  = os.path.join(args.output, "entities")
    relations_dir = os.path.join(args.output, "relations")
    os.makedirs(entities_dir,  exist_ok=True)
    os.makedirs(relations_dir, exist_ok=True)

    print("Converting entities...")
    convert_entities(args.entities, entities_dir)

    print("Converting relations...")
    convert_relations(args.relations, relations_dir)

    print(f"\nDone. Import files ready in {args.output}/")


if __name__ == "__main__":
    main()
