# -*- coding: utf-8 -*-
"""
Wipe all nodes and relationships from a Neo4j database.
Use before re-importing to start with a clean graph.

Usage:
    python wipe_kg.py --uri bolt://localhost:7687 --user neo4j --password <password>
    python wipe_kg.py --uri bolt://localhost:7687 --user neo4j --password <password> --dry-run

    Or set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in a .env file.
"""

import argparse
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Wipe all nodes and relationships from Neo4j.")
    parser.add_argument("--uri",      default=os.getenv("NEO4J_URI"))
    parser.add_argument("--user",     default=os.getenv("NEO4J_USERNAME"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print what would be deleted without actually deleting.")
    args = parser.parse_args()

    if not all([args.uri, args.user, args.password]):
        raise ValueError(
            "Neo4j credentials missing. Provide --uri, --user, --password "
            "or set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env"
        )

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as session:
            count = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
            print(f"Database at {args.uri} contains {count} node(s).")

            if args.dry_run:
                print("[DRY RUN] No changes made.")
                return

            session.run("MATCH (n) DETACH DELETE n")
            print("All nodes and relationships deleted.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
