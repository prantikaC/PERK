# -*- coding: utf-8 -*-
"""
Ingest per-type entity and relation CSVs into Neo4j to construct PERK.
Reads connection details from CLI args or environment variables.

Usage:
    python build_perk.py \
        --data_dir  neo4j_import/ \
        --ontology  perk_ontology.json \
        --uri       bolt://localhost:7687 \
        --user      neo4j \
        --password  <password>

    Or set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in a .env file.
"""

import argparse
import json
import os

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def create_nodes(tx, label, id_col, batch):
    query = f"""
    UNWIND $batch AS row
    MERGE (n:{label} {{id: row['{id_col}']}})
    SET n += row.properties
    """
    tx.run(query, batch=batch)


def create_relationships(tx, rel_type, batch):
    query = f"""
    UNWIND $batch AS row
    MATCH (source {{id: row['start_id']}})
    MATCH (target {{id: row['end_id']}})
    MERGE (source)-[r:{rel_type}]->(target)
    SET r += row.properties
    """
    tx.run(query, batch=batch)


def create_index(tx):
    tx.run("CREATE INDEX entity_id_index IF NOT EXISTS FOR (n:Entity) ON (n.id)")


def main():
    parser = argparse.ArgumentParser(description="Ingest PERK CSVs into Neo4j.")
    parser.add_argument("--data_dir",  required=True,
                        help="Directory containing entities/ and relations/ subdirs "
                             "(output of prepare_import.py)")
    parser.add_argument("--ontology",  default="perk_ontology.json")
    parser.add_argument("--uri",       default=os.getenv("NEO4J_URI"),
                        help="Neo4j bolt URI (or set NEO4J_URI)")
    parser.add_argument("--user",      default=os.getenv("NEO4J_USERNAME"),
                        help="Neo4j username (or set NEO4J_USERNAME)")
    parser.add_argument("--password",  default=os.getenv("NEO4J_PASSWORD"),
                        help="Neo4j password (or set NEO4J_PASSWORD)")
    args = parser.parse_args()

    if not all([args.uri, args.user, args.password]):
        raise ValueError(
            "Neo4j credentials missing. Provide --uri, --user, --password "
            "or set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env"
        )

    with open(args.ontology, "r") as f:
        ontology = json.load(f)

    entities_dir  = os.path.join(args.data_dir, "entities")
    relations_dir = os.path.join(args.data_dir, "relations")

    print(f"Connecting to Neo4j at {args.uri}...")
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    with driver.session() as session:
        session.execute_write(create_index)

        print("\n--- Ingesting Nodes ---")
        for entity_name, config in ontology["nodes"].items():
            file_path = os.path.join(entities_dir, config["file"])
            if not os.path.exists(file_path):
                print(f"  Skipping {entity_name}: {file_path} not found.")
                continue

            df = pd.read_csv(file_path)
            id_col        = config["id_col"]
            allowed_props = config.get("properties", [])
            batch = []

            for _, row in df.iterrows():
                row_dict = row.to_dict()
                node_id  = row_dict.get(id_col)
                if pd.isnull(node_id):
                    continue
                props = {p: row_dict[p] for p in allowed_props
                         if p in row_dict and pd.notnull(row_dict[p])}
                batch.append({id_col: node_id, "properties": props})

            if batch:
                session.execute_write(create_nodes, entity_name, id_col, batch)
                print(f"  {entity_name}: {len(batch)} nodes inserted.")

        print("\n--- Ingesting Relationships ---")
        for rel_name, config in ontology["relationships"].items():
            file_path = os.path.join(relations_dir, config["file"])
            if not os.path.exists(file_path):
                print(f"  Skipping {rel_name}: {file_path} not found.")
                continue

            df    = pd.read_csv(file_path)
            batch = []
            prop_cols = [c for c in df.columns if c not in [':START_ID', ':END_ID', ':TYPE']]

            for _, row in df.iterrows():
                batch.append({
                    "start_id":   row[':START_ID'],
                    "end_id":     row[':END_ID'],
                    "properties": {c: row[c] for c in prop_cols if pd.notnull(row[c])},
                })

            if batch:
                session.execute_write(create_relationships, config["type"], batch)
                print(f"  {rel_name}: {len(batch)} relationships inserted.")

    driver.close()
    print("\nPERK graph construction complete.")


if __name__ == "__main__":
    main()
