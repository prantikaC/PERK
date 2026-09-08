# -*- coding: utf-8 -*-
"""
Restore two Dataset nodes confirmed lost during entity resolution (found
while tracing "empty answer, gold has content" failures): entity fusion
merges two Dataset nodes it judges as duplicates by keeping only the
canonical node's name (node_fusion.py's `{**obs_p, **can_p}` property merge,
canonical always wins on key collision) -- the other node's distinct alias
is discarded entirely, not retained. Confirmed via raw pre-fusion extraction:

  - d12 "Chronicling America subset (1890-1900)" (entities_email13.csv) --
    absent from the live graph under any name/case; only the generic
    "Chronicling America" / "1890s datasets" nodes survived.
  - d22 "clean corpus subset" (entities_email25.csv) -- absent from the live
    graph entirely; likely merged into "clean subset of the corpus" or
    similar and lost its own name.

This is a narrow, manual restore for these two specific, individually-traced
cases (found via src/evaluation/kg_qa.py + deterministic_eval.py output
analysis), not a general fix for the underlying node_fusion.py bug -- a
systemic audit for other such losses is a separate, larger task.

Usage (dry-run by default):
    python restore_lost_datasets.py --model gpt
Apply:
    python restore_lost_datasets.py --model gpt --apply
"""

import argparse
import os

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()


def connect(prefix):
    uri = os.getenv(f"{prefix}_NEO4J_URI")
    user = os.getenv(f"{prefix}_NEO4J_USERNAME")
    pwd = os.getenv(f"{prefix}_NEO4J_PASSWORD")
    if not all([uri, user, pwd]):
        raise ValueError(f"Missing {prefix}_NEO4J_URI/USERNAME/PASSWORD in .env")
    return Neo4jGraph(url=uri, username=user, password=pwd)


RESTORES = [
    {
        "dataset_id": "restored_d12",
        "dataset_name": "Chronicling America subset (1890–1900)",
        "person_worksWith": ["Sunita Sen", "Dr. Ananya Chatterjee", "Dr. Michael Bradley"],
        # NOTE: raw extraction also linked this dataset to a Method named
        # "OCR correction" (me40:uses->d12), but that Method node does not
        # exist under any name/case in the live graph either -- it's a
        # THIRD lost node, out of scope for this restore. Skipping that
        # edge rather than inventing a new Method node here.
        "method_uses": [],
        "task_usedFor": [],
    },
    {
        "dataset_id": "restored_d22",
        "dataset_name": "clean corpus subset",
        "person_worksWith": ["Dr. Ananya Chatterjee"],
        "method_uses": ["LIME", "SHAP"],
        "task_usedFor": ["comparative evaluation of LIME and SHAP on clean corpus subset"],
    },
]


def main():
    p = argparse.ArgumentParser(description="Restore Dataset nodes lost during entity resolution.")
    p.add_argument("--model", required=True)
    p.add_argument("--apply", action="store_true", help="Write changes (default: dry-run only)")
    args = p.parse_args()

    g = connect(args.model.upper())
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] restoring lost Dataset nodes on {args.model.upper()} graph\n")

    for spec in RESTORES:
        existing = g.query(
            "MATCH (d:Dataset) WHERE toLower(d.datasetName) = toLower($name) RETURN d",
            params={"name": spec["dataset_name"]}
        )
        if existing:
            print(f"SKIP {spec['dataset_name']!r}: already exists in graph ({existing})")
            continue

        print(f"{spec['dataset_name']!r}:")
        person_matches = g.query(
            "MATCH (p:Person) WHERE p.personName IN $names RETURN p.personName AS name",
            params={"names": spec["person_worksWith"]}
        )
        method_matches = g.query(
            "MATCH (m:Method) WHERE m.methodName IN $names RETURN m.methodName AS name",
            params={"names": spec["method_uses"]}
        )
        task_matches = g.query(
            "MATCH (t:Task) WHERE t.taskName IN $names RETURN t.taskName AS name",
            params={"names": spec["task_usedFor"]}
        ) if spec["task_usedFor"] else []

        print(f"  Person(worksWith) matches found: {[r['name'] for r in person_matches]} "
              f"(expected {spec['person_worksWith']})")
        print(f"  Method(uses) matches found:      {[r['name'] for r in method_matches]} "
              f"(expected {spec['method_uses']})")
        if spec["task_usedFor"]:
            print(f"  Task(usedFor) matches found:     {[r['name'] for r in task_matches]} "
                  f"(expected {spec['task_usedFor']})")

        if args.apply:
            g.query(
                "CREATE (d:Dataset {id: $id, datasetName: $name})",
                params={"id": spec["dataset_id"], "name": spec["dataset_name"]}
            )
            for pname in spec["person_worksWith"]:
                g.query(
                    "MATCH (p:Person {personName: $pname}), (d:Dataset {id: $id}) "
                    "MERGE (p)-[:worksWith]->(d)",
                    params={"pname": pname, "id": spec["dataset_id"]}
                )
            for mname in spec["method_uses"]:
                g.query(
                    "MATCH (m:Method {methodName: $mname}), (d:Dataset {id: $id}) "
                    "MERGE (m)-[:uses]->(d)",
                    params={"mname": mname, "id": spec["dataset_id"]}
                )
            for tname in spec["task_usedFor"]:
                g.query(
                    "MATCH (d:Dataset {id: $id}), (t:Task {taskName: $tname}) "
                    "MERGE (d)-[:usedFor]->(t)",
                    params={"id": spec["dataset_id"], "tname": tname}
                )
            print(f"  -> created with {len(spec['person_worksWith'])} worksWith, "
                  f"{len(spec['method_uses'])} uses, {len(spec['task_usedFor'])} usedFor edges")
        print()

    if not args.apply:
        print("Re-run with --apply to create these nodes and relationships.")


if __name__ == "__main__":
    main()
