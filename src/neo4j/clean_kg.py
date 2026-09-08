# -*- coding: utf-8 -*-
"""
Validate entities and relations against PERKOnto before Neo4j import.
Strips invalid entity types, removes disallowed properties, and drops
relations whose (start_type, relation, end_type) triplet is not in the ontology.

Usage:
    python clean_kg.py \
        --entities_in  entities_fused.csv \
        --relations_in relations_fused.csv \
        --ontology     perk_ontology.json \
        --entities_out entities_clean.csv \
        --relations_out relations_clean.csv
"""

import argparse
import ast
import json
import logging
from collections import Counter

import pandas as pd

logger = logging.getLogger("clean_kg")


def sanitize_properties(row, valid_nodes, dropped_counts):
    allowed = set(valid_nodes[row['type']].get("properties", []))
    props_str = row['properties']
    try:
        if pd.isna(props_str):
            return "{}"
        try:
            props = json.loads(props_str)
        except json.JSONDecodeError:
            props = ast.literal_eval(props_str)
        if not isinstance(props, dict):
            return "{}"
        dropped = [k for k in props if k not in allowed]
        for k in dropped:
            dropped_counts[(row['type'], k)] += 1
        if dropped:
            logger.debug(
                f"Entity {row.get('id', '?')} ({row['type']}): stripped "
                f"non-ontology properties {dropped}"
            )
        return json.dumps({k: v for k, v in props.items() if k in allowed})
    except Exception:
        return "{}"


def is_valid_triplet(row, valid_relationships):
    rel = row['relation']
    if rel not in valid_relationships:
        return False
    return [row['start_type'], row['end_type']] in valid_relationships[rel].get("valid_pairs", [])


def main():
    parser = argparse.ArgumentParser(
        description="Validate KG data against PERKOnto before Neo4j import."
    )
    parser.add_argument("--entities_in",   required=True)
    parser.add_argument("--relations_in",  required=True)
    parser.add_argument("--ontology",      default="PERKOnto.json")
    parser.add_argument("--entities_out",  required=True)
    parser.add_argument("--relations_out", required=True)
    parser.add_argument("--log_file", default="clean_kg.log",
                         help="Per-row detail on everything dropped/stripped (DEBUG level).")
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(args.log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(file_handler)

    with open(args.ontology, "r") as f:
        ontology = json.load(f)

    valid_nodes         = ontology.get("nodes", {})
    valid_relationships = ontology.get("relationships", {})

    df_ent = pd.read_csv(args.entities_in)
    df_rel = pd.read_csv(args.relations_in)

    # Clean entities
    invalid_type_mask = ~df_ent['type'].isin(valid_nodes)
    if invalid_type_mask.any():
        for _, r in df_ent[invalid_type_mask].iterrows():
            logger.debug(f"Entity {r.get('id', '?')}: dropped -- unknown type '{r['type']}'")
        dropped_type_counts = Counter(df_ent.loc[invalid_type_mask, 'type'])
        print(f"Entities : dropped {invalid_type_mask.sum()} with unknown type: "
              f"{dict(dropped_type_counts)}")

    df_ent_clean = df_ent[~invalid_type_mask].copy()
    dropped_prop_counts = Counter()
    df_ent_clean['properties'] = df_ent_clean.apply(
        sanitize_properties, axis=1, valid_nodes=valid_nodes, dropped_counts=dropped_prop_counts
    )
    if dropped_prop_counts:
        print(f"Entities : stripped non-ontology properties (see {args.log_file} for detail): "
              f"{dict(dropped_prop_counts)}")
    print(f"Entities : {len(df_ent_clean)} / {len(df_ent)} retained")

    # Clean relations
    id_to_type = dict(zip(df_ent_clean['id'].astype(str), df_ent_clean['type']))
    df_rel_clean = df_rel.copy()
    df_rel_clean['start_id'] = df_rel_clean['start_id'].astype(str)
    df_rel_clean['end_id']   = df_rel_clean['end_id'].astype(str)
    df_rel_clean['start_type'] = df_rel_clean['start_id'].map(id_to_type)
    df_rel_clean['end_type']   = df_rel_clean['end_id'].map(id_to_type)

    dangling_mask = df_rel_clean['start_type'].isna() | df_rel_clean['end_type'].isna()
    if dangling_mask.any():
        for _, r in df_rel_clean[dangling_mask].iterrows():
            logger.debug(
                f"Relation {r['relation']} ({r['start_id']} -> {r['end_id']}): dropped -- "
                f"endpoint not a retained entity"
            )
        print(f"Relations: dropped {dangling_mask.sum()} with a dangling/dropped endpoint")
    df_rel_clean = df_rel_clean[~dangling_mask]

    valid_triplet_mask = df_rel_clean.apply(
        is_valid_triplet, axis=1, valid_relationships=valid_relationships
    )
    if (~valid_triplet_mask).any():
        for _, r in df_rel_clean[~valid_triplet_mask].iterrows():
            logger.debug(
                f"Relation {r['relation']} ({r['start_type']} -> {r['end_type']}) "
                f"[{r['start_id']} -> {r['end_id']}]: dropped -- not a valid ontology pair"
            )
        dropped_rel_counts = Counter(
            (r['relation'], r['start_type'], r['end_type'])
            for _, r in df_rel_clean[~valid_triplet_mask].iterrows()
        )
        print(f"Relations: dropped {(~valid_triplet_mask).sum()} with an invalid "
              f"(relation, start_type, end_type) triplet: {dict(dropped_rel_counts)}")
    df_rel_clean = df_rel_clean[valid_triplet_mask]
    df_rel_clean = df_rel_clean.drop(columns=['start_type', 'end_type'])
    print(f"Relations: {len(df_rel_clean)} / {len(df_rel)} retained")

    df_ent_clean.to_csv(args.entities_out, index=False)
    df_rel_clean.to_csv(args.relations_out, index=False)
    print(f"Saved to {args.entities_out} and {args.relations_out}")
    print(f"Full per-row drop/strip detail logged to {args.log_file}")


if __name__ == "__main__":
    main()
