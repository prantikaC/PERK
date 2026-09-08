# -*- coding: utf-8 -*-
"""
Export the PERK Neo4j knowledge graph as (subject, predicate, object) triples,
written to BOTH a JSON file and a CSV file.

Usage:
    python export_perk_triples.py --model gpt --output PERK_triples
        -> writes PERK_triples.json and PERK_triples.csv

Credentials are read from env vars (same convention as kg_eval.py):
    <MODEL_UPPER>_NEO4J_URI, <MODEL_UPPER>_NEO4J_USERNAME, <MODEL_UPPER>_NEO4J_PASSWORD
    e.g. --model gpt  =>  GPT_NEO4J_URI, GPT_NEO4J_USERNAME, GPT_NEO4J_PASSWORD

Triple kinds emitted (each row is one triple):
    relation  : (subject_entity) -[predicate]-> (object_entity)   object_kind=entity
    type      : (subject_entity) -[type]->       Label            object_kind=class   (--no-types to skip)
    attribute : (subject_entity) -[prop_name]->  literal_value    object_kind=literal (--no-attributes to skip)

Each triple carries id + human label + type on both sides so it round-trips and
stays readable:
    subject, subject_label, subject_type,
    predicate,
    object, object_label, object_type, object_kind
"""

import argparse
import csv
import json
import os
from collections import Counter

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

# per-type property that holds the human-readable name of a node
TYPE_LABEL_KEY = {
    "Person":       "personName",
    "Task":         "taskName",
    "Paper":        "paperTitle",
    "Conference":   "confTitle",
    "Journal":      "journalTitle",
    "Dataset":      "datasetName",
    "Method":       "methodName",
    "Metric":       "metricName",
    "Meeting":      "meetAgenda",
    "Email":        "subject",
    "PaperStatus":  "statusType",
    "SubmissionID": "identifier",
}

# generic fallbacks (any type), tried if the type-specific key is missing
LABEL_KEYS = [
    "personName", "taskName", "paperTitle", "confTitle", "journalTitle",
    "datasetName", "methodName", "metricName", "meetAgenda", "subject",
    "statusType", "identifier", "name", "title", "label",
]

CSV_FIELDS = [
    "subject", "subject_label", "subject_type",
    "predicate",
    "object", "object_label", "object_type", "object_kind",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Export a PERK Neo4j KG as subject-predicate-object triples (JSON + CSV)."
    )
    p.add_argument(
        "--model", required=True,
        help="KG prefix used to look up env vars, e.g. 'gpt' -> GPT_NEO4J_URI",
    )
    p.add_argument(
        "--output", default="PERK_triples",
        help="Output basename; writes <output>.json and <output>.csv",
    )
    p.add_argument("--no-types", action="store_true",
                   help="Do not emit (entity)-[type]->Label triples")
    p.add_argument("--no-attributes", action="store_true",
                   help="Do not emit (entity)-[prop]->literal triples")
    p.add_argument("--indent", type=int, default=2, help="JSON indent (0 = compact)")
    return p.parse_args()


def connect_neo4j(prefix):
    uri  = os.getenv(f"{prefix}_NEO4J_URI")
    user = os.getenv(f"{prefix}_NEO4J_USERNAME")
    pwd  = os.getenv(f"{prefix}_NEO4J_PASSWORD")
    if not all([uri, user, pwd]):
        raise ValueError(
            f"Missing credentials for prefix '{prefix}'. "
            f"Set {prefix}_NEO4J_URI, {prefix}_NEO4J_USERNAME, {prefix}_NEO4J_PASSWORD in .env"
        )
    return Neo4jGraph(url=uri, username=user, password=pwd)


def primary_type(labels):
    """Domain label, ignoring the generic :Entity marker."""
    domain = [l for l in labels if l != "Entity"]
    return domain[0] if domain else (labels[0] if labels else None)


def display_label(props, node_id, node_type=None):
    # 1. type-specific name property
    key = TYPE_LABEL_KEY.get(node_type)
    if key and props.get(key) not in (None, ""):
        return str(props[key])
    # 2. any known name-ish property
    for k in LABEL_KEYS:
        v = props.get(k)
        if v not in (None, ""):
            return str(v)
    # 3. last resort: the id
    return str(node_id)


def fetch_nodes(graph):
    rows = graph.query(
        "MATCH (n) "
        "RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties "
        "ORDER BY id"
    )
    nodes = {}
    for r in rows:
        props = dict(r["properties"] or {})
        ntype = primary_type(r["labels"])
        nodes[r["id"]] = {
            "id": r["id"],
            "type": ntype,
            "label": display_label(props, r["id"], ntype),
            "properties": props,
        }
    return nodes


def fetch_relations(graph):
    return graph.query(
        "MATCH (a)-[r]->(b) "
        "RETURN a.id AS source, type(r) AS type, b.id AS target "
        "ORDER BY source, type, target"
    )


def build_triples(nodes, relations, include_types, include_attributes):
    triples = []

    # relation triples: entity -> entity
    for r in relations:
        s = nodes.get(r["source"])
        o = nodes.get(r["target"])
        triples.append({
            "subject":       r["source"],
            "subject_label": s["label"] if s else r["source"],
            "subject_type":  s["type"]  if s else None,
            "predicate":     r["type"],
            "object":        r["target"],
            "object_label":  o["label"] if o else r["target"],
            "object_type":   o["type"]  if o else None,
            "object_kind":   "entity",
        })

    # type + attribute triples: entity -> literal/class
    for n in nodes.values():
        if include_types and n["type"] is not None:
            triples.append({
                "subject":       n["id"],
                "subject_label": n["label"],
                "subject_type":  n["type"],
                "predicate":     "type",
                "object":        n["type"],
                "object_label":  n["type"],
                "object_type":   "Class",
                "object_kind":   "class",
            })
        if include_attributes:
            for k, v in n["properties"].items():
                if k == "id" or v in (None, ""):
                    continue
                triples.append({
                    "subject":       n["id"],
                    "subject_label": n["label"],
                    "subject_type":  n["type"],
                    "predicate":     k,
                    "object":        v,
                    "object_label":  str(v),
                    "object_type":   "Literal",
                    "object_kind":   "literal",
                })
    return triples


def main():
    args = parse_args()
    prefix = args.model.upper()

    graph = connect_neo4j(prefix)
    print(f"Connected to {prefix} KG. Exporting triples...")

    nodes = fetch_nodes(graph)
    relations = fetch_relations(graph)
    triples = build_triples(
        nodes, relations,
        include_types=not args.no_types,
        include_attributes=not args.no_attributes,
    )

    kind_counts = Counter(t["object_kind"] for t in triples)
    pred_counts = Counter(t["predicate"] for t in triples if t["object_kind"] == "entity")

    payload = {
        "metadata": {
            "source_prefix": prefix,
            "num_entities": len(nodes),
            "num_triples": len(triples),
            "triples_by_kind": dict(kind_counts),
            "relation_predicates": dict(pred_counts),
        },
        "triples": triples,
    }

    json_path = f"{args.output}.json"
    csv_path = f"{args.output}.csv"

    indent = args.indent if args.indent > 0 else None
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for t in triples:
            writer.writerow(t)

    print(f"Wrote {len(triples)} triples ({len(nodes)} entities) to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"Triples by kind:      {dict(kind_counts)}")
    print(f"Relation predicates:  {dict(pred_counts)}")


if __name__ == "__main__":
    main()
