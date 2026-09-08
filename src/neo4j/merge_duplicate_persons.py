# -*- coding: utf-8 -*-
"""
Merge known duplicate Person nodes into their canonical node on a live PERK graph.

Clusters identified by exact-normalized-name and fuzzy-name matching across all
66 Person nodes, then cross-checked against the authoritative participant list
in src/prompts/patra_gen_prompt.txt to pick the canonical node in each cluster
(the one whose email matches the prompt's fixed roster, not just whichever
node happens to have the most edges already).

Usage (dry-run by default):
    python merge_duplicate_persons.py --model gpt
Apply:
    python merge_duplicate_persons.py --model gpt --apply
"""

import argparse
import os

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv(override=True)

# duplicate_id -> canonical_id
#
# --- gpt (older graph) mapping -- kept for reference, do not use against gpt_v5 ---
# Ramesh: pn19 'Ramesh Bhatia' (no email) -> pn2 'Prof. Ramesh Bhatia' (matches prompt email)
# Ananya: pn4/pn8/pn15 -> pn9 (matches prompt email ananya.chatterjee@cs.ox.ac.uk exactly; pn8's
#         "ananya.chatterjee@ox.ac.uk" is itself a drift variant missing "cs.", NOT canonical)
# Bradley: pn3 'Dr Michael Bradly' (misspelled) -> pn7 'Dr. Michael Bradley' (matches prompt email)
#
# --- gpt_v5 (current graph) mapping -- different extraction run, different ids ---
# Bradley: pn7 'Michael Bradly' (misspelled, no email) -> pn3 'Michael Bradley' (owns eid3).
#          This is a clear spelling-typo fix, not an identity judgment call.
# pn8 'Dr Bradley' was PREVIOUSLY also mapped to pn3 here, on my own unverified
# assumption that it must be the same person -- no actual evidence (context
# quotes, email ownership, etc.) was checked before making that call. Reverted
# per explicit correction: do not merge pn8 into pn3 without real verification.
MERGE_MAP = {
    "pn7": "pn3",
}


def connect(prefix):
    uri = os.getenv(f"{prefix}_NEO4J_URI")
    user = os.getenv(f"{prefix}_NEO4J_USERNAME")
    pwd = os.getenv(f"{prefix}_NEO4J_PASSWORD")
    if not all([uri, user, pwd]):
        raise ValueError(f"Missing {prefix}_NEO4J_URI/USERNAME/PASSWORD in .env")
    return Neo4jGraph(url=uri, username=user, password=pwd)


def main():
    p = argparse.ArgumentParser(description="Merge known duplicate Person nodes into their canonical node.")
    p.add_argument("--model", required=True)
    p.add_argument("--apply", action="store_true", help="Write changes (default: dry-run only)")
    args = p.parse_args()

    g = connect(args.model.upper())
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] merging duplicate Person nodes on {args.model.upper()} graph\n")

    for dup_id, canon_id in MERGE_MAP.items():
        dup = g.query("MATCH (p:Person {id:$id}) RETURN p.personName AS name", params={"id": dup_id})
        canon = g.query("MATCH (p:Person {id:$id}) RETURN p.personName AS name", params={"id": canon_id})
        if not dup or not canon:
            print(f"  SKIP {dup_id} -> {canon_id}: node not found")
            continue
        dup_name, canon_name = dup[0]["name"], canon[0]["name"]

        out_rels = g.query(
            "MATCH (d:Person {id:$id})-[r]->(o) RETURN type(r) AS reltype, o.id AS other_id, properties(r) AS props",
            params={"id": dup_id})
        in_rels = g.query(
            "MATCH (o)-[r]->(d:Person {id:$id}) RETURN type(r) AS reltype, o.id AS other_id, properties(r) AS props",
            params={"id": dup_id})

        print(f"{dup_id} ({dup_name!r}) -> {canon_id} ({canon_name!r}): "
              f"{len(out_rels)} outgoing, {len(in_rels)} incoming relationships to redirect")

        if args.apply:
            for r in out_rels:
                g.query(
                    f"MATCH (c:Person {{id:$cid}}), (o {{id:$oid}}) "
                    f"MERGE (c)-[nr:`{r['reltype']}`]->(o) SET nr += $props",
                    params={"cid": canon_id, "oid": r["other_id"], "props": r["props"] or {}})
            for r in in_rels:
                g.query(
                    f"MATCH (o {{id:$oid}}), (c:Person {{id:$cid}}) "
                    f"MERGE (o)-[nr:`{r['reltype']}`]->(c) SET nr += $props",
                    params={"cid": canon_id, "oid": r["other_id"], "props": r["props"] or {}})
            g.query("MATCH (d:Person {id:$id}) DETACH DELETE d", params={"id": dup_id})
            print(f"  -> merged and deleted {dup_id}")

    if not args.apply:
        print("\nRe-run with --apply to perform the merges.")
    else:
        remaining = g.query("MATCH (p:Person) RETURN count(p) AS c")[0]["c"]
        print(f"\n[APPLY] Person nodes remaining: {remaining}")


if __name__ == "__main__":
    main()
