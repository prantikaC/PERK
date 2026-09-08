# -*- coding: utf-8 -*-
"""
Remove entities whose name is a bare, vague placeholder ("dataset", "task",
"method", "paper", ...) rather than any real, specific identifying name, on
a live PERK graph.

IMPORTANT: only the EXACT-match denylist (GENERIC_NAME_DENYLIST_EXACT in
kg_extraction_pipeline.py) is safe to use for deletion. That module also has
a broader SUBSTRING denylist ("submission", "the task", ...), but that one
was designed for DEDUP-KEY purposes only, where over-matching is harmless
(worst case: a real entity just never gets merged with anything). Using the
substring list for deletion is a different, much riskier bar -- confirmed
empirically it flags ~380 Task entities on the gpt_v5 graph, and most of
those are legitimate, specific tasks that merely contain a generic word
(e.g. "touch upon possible venues... for ultimate submission" is a real,
meaningful task, not a placeholder). Deleting those would destroy real data.
Exact-match only, by contrast, is a small set (3 on gpt_v5) of entities
whose ENTIRE name is just "dataset"/"task"/"paper"/etc. with nothing else --
unambiguously useless regardless of type.

Usage (dry-run by default):
    python remove_generic_entities.py --model gpt_v5
Apply:
    python remove_generic_entities.py --model gpt_v5 --apply
"""

import argparse
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)

GENERIC_NAME_DENYLIST_EXACT = {
    "dataset", "datasets", "document", "documents", "method", "methods",
    "metric", "metrics", "task", "tasks", "paper", "papers", "manuscript",
}

NAME_FIELD = {
    "Dataset": "datasetName", "Method": "methodName", "Task": "taskName",
    "Metric": "metricName", "Paper": "paperTitle",
}


def get_driver(prefix):
    prefix = prefix.upper()
    uri = os.getenv(f"{prefix}_NEO4J_URI")
    user = os.getenv(f"{prefix}_NEO4J_USERNAME")
    pwd = os.getenv(f"{prefix}_NEO4J_PASSWORD")
    if not all([uri, user, pwd]):
        raise ValueError(f"Missing {prefix}_NEO4J_URI/USERNAME/PASSWORD in .env")
    return GraphDatabase.driver(uri, auth=(user, pwd))


def main():
    parser = argparse.ArgumentParser(
        description="Remove entities with bare generic-placeholder names (exact match only)."
    )
    parser.add_argument("--model", required=True, help="Prefix used to look up env vars, e.g. 'gpt_v5'")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run only)")
    args = parser.parse_args()

    driver = get_driver(args.model)
    with driver.session() as session:
        found = []
        for label, field in NAME_FIELD.items():
            rows = session.run(
                f"MATCH (n:{label}) WHERE toLower(trim(n.{field})) IN $names "
                f"RETURN n.id AS id, n.{field} AS name",
                names=list(GENERIC_NAME_DENYLIST_EXACT),
            )
            for r in rows:
                found.append((label, r["id"], r["name"]))

        print(f"Found {len(found)} generic-placeholder entit(y/ies):")
        for label, eid, name in found:
            rel_count = session.run(
                "MATCH (n {id:$id})-[r]-() RETURN count(r) AS c", id=eid
            ).single()["c"]
            print(f"  {label} {eid} ({name!r}) -- {rel_count} relationship(s) attached")

        if not found:
            print("Nothing to do.")
            return

        if args.apply:
            for _, eid, _ in found:
                session.run("MATCH (n {id:$id}) DETACH DELETE n", id=eid)
            print(f"\nDeleted {len(found)} entit(y/ies) and all their relationships.")
        else:
            print(f"\n[DRY-RUN] Would delete {len(found)} entit(y/ies) and all their relationships. "
                  f"Re-run with --apply to perform the deletion.")

    driver.close()


if __name__ == "__main__":
    main()
