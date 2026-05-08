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

import pandas as pd


def sanitize_properties(row, valid_nodes):
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
    args = parser.parse_args()

    with open(args.ontology, "r") as f:
        ontology = json.load(f)

    valid_nodes         = ontology.get("nodes", {})
    valid_relationships = ontology.get("relationships", {})

    df_ent = pd.read_csv(args.entities_in)
    df_rel = pd.read_csv(args.relations_in)

    # Clean entities
    df_ent_clean = df_ent[df_ent['type'].isin(valid_nodes)].copy()
    df_ent_clean['properties'] = df_ent_clean.apply(
        sanitize_properties, axis=1, valid_nodes=valid_nodes
    )
    print(f"Entities : {len(df_ent_clean)} / {len(df_ent)} retained")

    # Clean relations
    id_to_type = dict(zip(df_ent_clean['id'].astype(str), df_ent_clean['type']))
    df_rel_clean = df_rel.copy()
    df_rel_clean['start_id'] = df_rel_clean['start_id'].astype(str)
    df_rel_clean['end_id']   = df_rel_clean['end_id'].astype(str)
    df_rel_clean['start_type'] = df_rel_clean['start_id'].map(id_to_type)
    df_rel_clean['end_type']   = df_rel_clean['end_id'].map(id_to_type)

    df_rel_clean = df_rel_clean[
        df_rel_clean['start_type'].notna() & df_rel_clean['end_type'].notna()
    ]
    df_rel_clean = df_rel_clean[
        df_rel_clean.apply(is_valid_triplet, axis=1, valid_relationships=valid_relationships)
    ]
    df_rel_clean = df_rel_clean.drop(columns=['start_type', 'end_type'])
    print(f"Relations: {len(df_rel_clean)} / {len(df_rel)} retained")

    df_ent_clean.to_csv(args.entities_out, index=False)
    df_rel_clean.to_csv(args.relations_out, index=False)
    print(f"Saved to {args.entities_out} and {args.relations_out}")


if __name__ == "__main__":
    main()
