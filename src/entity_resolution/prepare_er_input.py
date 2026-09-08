# -*- coding: utf-8 -*-
"""
Merge header-extracted entities/relations (header_signature_parser.py output)
with body-extracted entities/relations (filter_body_entities.py output) into
a single entities_final.csv/relations_final.csv pair ready for the existing
entity-resolution pipeline (faiss_blocking.py -> llm_judgement.py -> node_fusion.py).

Why this is needed rather than just concatenating the two input files:

  1. ID collision: header_signature_parser.py and the body extraction
     pipeline each run their own independent id counters, and can land on
     the same prefix+number for two different entities purely by chance
     (e.g. header's pn3 = Michael Bradley, body's own pn3 = some unrelated
     person). filter_body_entities.py already removes the *systematic*
     collision (the same real person getting a body-side Person node
     redundant with a header-side one, joined via personEmail); what's left
     is coincidental leftover overlap on ids neither script has any reason
     to know about. Header ids are kept exactly as-is; any body entity id
     whose prefix also appears in the header set gets renumbered to start
     after the header's highest number for that prefix (e.g. if header uses
     pn1..pn30, body's own pn1 becomes pn31, pn2 becomes pn32, ...,
     preserving body's original relative order). Prefixes that don't appear
     in the header set (Paper, Task, Dataset, ...) are left untouched.
  2. Schema mismatch: header_relations.csv is 9 columns (start_id,
     start_type, start_label, relation, end_id, end_type, end_label,
     relProperties, source_mailNums); the ER pipeline (and body_relations.csv)
     expects 5 (start_id, end_id, relation, context, source). Header
     relations have no free-text quote to use as `context`, so one is
     synthesized from the start/end labels and relProperties -- this is what
     faiss_blocking.py embeds as evidence text for Person/Team/Organization
     resolution.
  3. Ontology-hallucination filtering: LLM body extraction produces some
     relations whose (start_type, relation, end_type) triplet is not in
     PERKOnto.json at all (e.g. movesTo landing on a Task or a Paper instead
     of PaperStatus) -- previously these only got caught by clean_kg.py at
     the very end, after already influencing FAISS blocking context and
     LLM-judgment prompts for whatever they touched. They are dropped here,
     before resolution, so they can never affect which entities look
     similar to each other. Only the invalid relation is dropped -- the
     entities on either end are kept, since they may have other valid
     relations.

PaperStatus/Email/MailThread/EmailID need no special-casing to stay out of
resolution: their properties (statusType/mailNum/mailDate/threadID/eID) are
simply not in faiss_blocking.py's TARGET_KEYS, so they never become
resolution candidates in the first place. Team/Organization DO need
'teamName'/'orgName' in TARGET_KEYS for resolution to consider them at all --
see the accompanying edit to faiss_blocking.py.

Usage:
    python prepare_er_input.py \
        --header_entities  data/header_extractions/openai_v2/header_entities.csv \
        --header_relations data/header_extractions/openai_v2/header_relations.csv \
        --body_entities    data/body_extractions/openai_v2/body_entities.csv \
        --body_relations   data/body_extractions/openai_v2/body_relations.csv \
        --ontology         ontology/PERKOnto.json \
        --output_dir       data/entity_resolution/openai_v2 \
        --prefix           openai_v2
"""

import argparse
import csv
import json
import os
import re
from collections import Counter

ENTITY_FIELDS = ["id", "type", "properties"]
# role/date: header_signature_parser.py already extracts these onto
# memberOf/affiliation relations (parsed from the sender's own signature
# block) into a relProperties JSON blob -- previously this schema didn't
# include them, so they got flattened into unstructured `context` text and
# never reached Neo4j as real, queryable relationship properties. Declared
# on memberOf/affiliation in PERKOnto.json; body relations never populate
# them (extraction_prompt.txt doesn't ask for role/date), so they're empty
# there.
RELATION_FIELDS = ["start_id", "end_id", "relation", "context", "source", "role", "date"]
ID_RE = re.compile(r'^([A-Za-z]+)(\d+)$')


def load_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def split_id(entity_id):
    m = ID_RE.match(entity_id)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def clean_header(header_entities, header_relations):
    entities = [
        {"id": row['id'], "type": row['type'], "properties": row['properties']}
        for row in header_entities
    ]
    relations = []
    for row in header_relations:
        rel_props_raw = (row.get('relProperties') or '').strip()
        rel_props = {}
        if rel_props_raw and rel_props_raw != '{}':
            try:
                rel_props = json.loads(rel_props_raw)
            except (json.JSONDecodeError, TypeError):
                rel_props = {}
        context_bits = [f"{row['start_label']} {row['relation']} {row['end_label']}"]
        relations.append({
            "start_id": row['start_id'],
            "end_id":   row['end_id'],
            "relation": row['relation'],
            "context":  " | ".join(context_bits),
            "source":   row.get('source_mailNums', ''),
            "role":     rel_props.get('role', ''),
            "date":     rel_props.get('date', ''),
        })
    return entities, relations


def clean_body(body_entities, body_relations):
    entities = [
        {"id": row['id'], "type": row['type'], "properties": row['properties']}
        for row in body_entities
    ]
    relations = [
        {"start_id": row['start_id'], "end_id": row['end_id'], "relation": row['relation'],
         "context": row.get('context', ''), "source": row.get('source', ''),
         "role": "", "date": ""}
        for row in body_relations
    ]
    return entities, relations


def renumber_colliding_body_ids(header_entities, body_entities, body_relations):
    """Body ids whose prefix also appears in the header namespace get pushed
    past the header's max for that prefix, in their original body order.
    Body ids whose prefix never appears in the header set are untouched."""
    header_max = {}
    for row in header_entities:
        prefix, num = split_id(row['id'])
        if prefix is not None:
            header_max[prefix] = max(header_max.get(prefix, 0), num)

    id_map = {}
    next_num = dict(header_max)
    for row in body_entities:
        prefix, num = split_id(row['id'])
        if prefix is not None and prefix in header_max:
            next_num[prefix] += 1
            id_map[row['id']] = f"{prefix}{next_num[prefix]}"

    entities = [
        {"id": id_map.get(row['id'], row['id']), "type": row['type'], "properties": row['properties']}
        for row in body_entities
    ]
    relations = [
        {"start_id": id_map.get(row['start_id'], row['start_id']),
         "end_id":   id_map.get(row['end_id'], row['end_id']),
         "relation": row['relation'], "context": row['context'], "source": row['source'],
         "role": row.get('role', ''), "date": row.get('date', '')}
        for row in body_relations
    ]
    return entities, relations, id_map


def filter_ontology_violations(relations, id_to_type, ontology):
    valid_relationships = ontology.get("relationships", {})
    kept, dropped = [], []
    for row in relations:
        spec = valid_relationships.get(row['relation'])
        pair = [id_to_type.get(row['start_id']), id_to_type.get(row['end_id'])]
        if spec and pair in spec.get('valid_pairs', []):
            kept.append(row)
        else:
            dropped.append((row['relation'], pair[0], pair[1]))
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(
        description="Merge header + body extraction output into ER-pipeline input, "
                     "dropping ontology-invalid relation triplets before resolution."
    )
    parser.add_argument("--header_entities", required=True)
    parser.add_argument("--header_relations", required=True)
    parser.add_argument("--body_entities", required=True)
    parser.add_argument("--body_relations", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", required=True, help="Output file prefix, e.g. 'openai_v2'")
    args = parser.parse_args()

    header_entities, header_relations = clean_header(
        load_rows(args.header_entities), load_rows(args.header_relations)
    )
    body_entities, body_relations = clean_body(
        load_rows(args.body_entities), load_rows(args.body_relations)
    )
    body_entities, body_relations, renumbered = renumber_colliding_body_ids(
        header_entities, body_entities, body_relations
    )

    all_entities = header_entities + body_entities
    all_relations = header_relations + body_relations

    seen, dup_ids = set(), set()
    for row in all_entities:
        if row['id'] in seen:
            dup_ids.add(row['id'])
        seen.add(row['id'])
    if dup_ids:
        raise SystemExit(f"ID collision after renumbering, aborting: {sorted(dup_ids)[:10]}")

    id_to_type = {row['id']: row['type'] for row in all_entities}
    with open(args.ontology) as f:
        ontology = json.load(f)
    kept_relations, dropped = filter_ontology_violations(all_relations, id_to_type, ontology)

    os.makedirs(args.output_dir, exist_ok=True)
    ent_out = os.path.join(args.output_dir, f"{args.prefix}_entities_final.csv")
    rel_out = os.path.join(args.output_dir, f"{args.prefix}_relations_final.csv")

    with open(ent_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ENTITY_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(all_entities)

    with open(rel_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RELATION_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(kept_relations)

    print(f"Entities: {len(all_entities)} total "
          f"({len(header_entities)} header + {len(body_entities)} body) -> {ent_out}")
    print(f"Relations: {len(kept_relations)} / {len(all_relations)} kept "
          f"({len(dropped)} dropped as ontology-invalid triplets) -> {rel_out}")
    if renumbered:
        by_prefix = Counter(split_id(v)[0] for v in renumbered.values())
        print(f"Body ids renumbered to avoid colliding with header ids: {len(renumbered)} {dict(by_prefix)}")
    if dropped:
        print(f"Dropped triplet breakdown: {dict(Counter(dropped))}")


if __name__ == "__main__":
    main()
