# -*- coding: utf-8 -*-
"""
Merge known duplicate Person nodes into their canonical node on the live
Qwen32B-PERK (qwen32b_v1) graph.

Clusters identified by exact-string personName match across all 35 Person
nodes on this graph, then cross-checked against the authoritative
participant list in src/prompts/patra_gen_prompt.txt (via hasOwner ->
EmailID.eID) to pick the canonical node in each cluster -- the one whose
owned email matches the prompt's fixed roster exactly, not just whichever
node happens to have the most edges already.

Sunita Sen:        pn1 (owns scholar.sunita@iacs.res.in, roster match) <- pn11
Ramesh Bhatia:      pn2 (owns ramesh.bhatia@iacs.res.in, roster match)  <- pn7, pn12
Michael Bradley:    pn3 (owns michael@cs.stanford.edu, roster match)    <- pn8
Ananya Chatterjee:  pn5 (owns ananya.chatterjee@cs.ox.ac.uk, exact roster
                     match) <- pn4, pn9, pn10
                     pn4 owns "ananya.chatterjee@ox.ac.uk" -- missing "cs.",
                     a drift variant, NOT canonical (same pattern seen on
                     the gpt/gpt_v5 graphs).

All other Person nodes on this graph are either genuinely distinct people
or informal/partial mentions ("Michael", "Ramesh", "Ms. Sen", "co-authors",
"expert in this field", ...) with no corroborating email evidence, and are
deliberately left unmerged rather than guessed.

Usage (dry-run by default):
    python merge_duplicate_persons_qwen32b_v1.py --model qwen32b_v1
Apply:
    python merge_duplicate_persons_qwen32b_v1.py --model qwen32b_v1 --apply
"""

import argparse
import os

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv(override=True)

# duplicate_id -> canonical_id
MERGE_MAP = {
    "pn11": "pn1",
    "pn7":  "pn2",
    "pn12": "pn2",
    "pn8":  "pn3",
    "pn4":  "pn5",
    "pn9":  "pn5",
    "pn10": "pn5",
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
