# -*- coding: utf-8 -*-
"""
Dump a LIVE PERK Neo4j graph into build_perk.py's import format
(entities/<Label>.csv + relations/<type>.csv), dropping any relation whose
(start_type, end_type) is not an ontology valid_pair (forward only).

Use this to derive an ontology-clean v2 that differs from the live graph by
*exactly* the violating edges, with all node properties preserved — unlike
re-filtering a stale on-disk import dir.

Usage:
    # v1 (PERK_GPT) must be the running instance on the connection
    python dump_import_from_live.py --model gpt \
        --ontology ontology/PERKOnto.json \
        --out_dir /path/to/live/neo4j_import

Then build into the (empty) v2 instance:
    python build_perk.py --data_dir <out_dir> --ontology ontology/PERKOnto.json \
        --uri bolt://localhost:7687 --user neo4j --password <V2_PW>
"""

import argparse
import csv
import json
import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Dump a live PERK graph to a clean build_perk import dir.")
    p.add_argument("--model", required=True, help="Env prefix, e.g. gpt -> GPT_NEO4J_URI")
    p.add_argument("--ontology", default="PERKOnto.json")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def connect(prefix):
    uri = os.getenv(f"{prefix}_NEO4J_URI")
    user = os.getenv(f"{prefix}_NEO4J_USERNAME")
    pwd = os.getenv(f"{prefix}_NEO4J_PASSWORD")
    if not all([uri, user, pwd]):
        raise ValueError(f"Missing {prefix}_NEO4J_URI/USERNAME/PASSWORD in .env")
    return Neo4jGraph(url=uri, username=user, password=pwd)


def main():
    args = parse_args()
    prefix = args.model.upper()
    onto = json.load(open(args.ontology, encoding="utf-8"))

    valid = defaultdict(set)
    for rn, rl in onto["relationships"].items():
        for pair in rl.get("valid_pairs", []):
            if len(pair) == 2:
                valid[rl.get("type", rn)].add((pair[0], pair[1]))

    g = connect(prefix)
    ent_dir = os.path.join(args.out_dir, "entities")
    rel_dir = os.path.join(args.out_dir, "relations")
    os.makedirs(ent_dir, exist_ok=True)
    os.makedirs(rel_dir, exist_ok=True)

    # ---- entities: one CSV per ontology node label, real properties from live graph
    print("--- Dumping nodes ---")
    for label, cfg in onto["nodes"].items():
        props = cfg.get("properties", [])
        rows = g.query(
            f"MATCH (n:`{label}`) RETURN n.id AS id, properties(n) AS p ORDER BY n.id"
        )
        fname = os.path.basename(cfg["file"])
        header = [":ID"] + props + [":LABEL"]
        with open(os.path.join(ent_dir, fname), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                p = r["p"] or {}
                w.writerow([r["id"]] + [p.get(k, "") for k in props] + [label])
        print(f"  {label}: {len(rows)} nodes")

    # ---- relations: one CSV per type, drop ontology-violating pairs
    print("--- Dumping relations (ontology-valid only) ---")
    total_kept = total_all = 0
    for rn, cfg in onto["relationships"].items():
        rtype = cfg.get("type", rn)
        rows = g.query(
            f"MATCH (a)-[r:`{rtype}`]->(b) "
            f"RETURN a.id AS s, b.id AS o, labels(a) AS sl, labels(b) AS ol, properties(r) AS p "
            f"ORDER BY a.id, b.id"
        )
        fname = os.path.basename(cfg["file"])
        kept = 0
        with open(os.path.join(rel_dir, fname), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([":START_ID", ":END_ID", ":TYPE", "context", "source"])
            for r in rows:
                s = next((x for x in r["sl"] if x != "Entity"), None)
                o = next((x for x in r["ol"] if x != "Entity"), None)
                if (s, o) in valid.get(rtype, set()):
                    p = r["p"] or {}
                    w.writerow([r["s"], r["o"], rtype, p.get("context", ""), p.get("source", "")])
                    kept += 1
        total_kept += kept
        total_all += len(rows)
        print(f"  {rtype:14s} {kept:6d} / {len(rows):6d}  (dropped {len(rows)-kept})")
    print(f"\nTOTAL relations kept: {total_kept} / {total_all}  (dropped {total_all-total_kept})")
    print(f"Wrote clean import to {args.out_dir}")


if __name__ == "__main__":
    main()
