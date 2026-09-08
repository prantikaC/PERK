# -*- coding: utf-8 -*-
"""
Convert master entities/relations CSVs into per-type CSVs
in Neo4j bulk import format.

Output structure:
    <output>/
    ├── entities/
    │   ├── Person.csv
    │   ├── Paper.csv
    │   └── ...
    └── relations/
        ├── sentBy.csv
        ├── hasAuthor.csv
        └── ...

Usage:
    python prepare_import.py \
        --entities  entities_clean.csv \
        --relations relations_clean.csv \
        --output    neo4j_import/
"""

import argparse
import ast
import csv
import os
import re
import warnings
from collections import Counter
from datetime import datetime

from dateutil import parser as _dateparser

# some date strings carry unhandled tz abbreviations (e.g. "16 June 2019 BST");
# the date still parses correctly, so quiet the noisy warning.
warnings.filterwarnings("ignore", module="dateutil")

# --- date normalization -------------------------------------------------------
# Any property whose key ends in "date" (mailDate, taskDate, meetDate, confDate,
# statusDate, ...) is normalized to ISO YYYY-MM-DD. Values that are not a fully
# specified calendar date (relative words like "tomorrow", partials like "2020-08"
# or "20 November", empty) are blanked. This is what lets Cypher call date() on
# these properties without throwing: date() on a real ISO string works, and a
# blanked value becomes a null property so date(null) returns null (not an error).
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_DEF_A = datetime(2000, 1, 1)
_DEF_B = datetime(2001, 6, 15)


def normalize_date(value):
    """Return ISO YYYY-MM-DD if `value` is a fully specified date, else ''."""
    s = str(value).strip()
    if not s:
        return ""
    if _ISO_DATE.match(s):
        return s[:10]
    try:
        a = _dateparser.parse(s, default=_DEF_A, fuzzy=True, dayfirst=True)
        b = _dateparser.parse(s, default=_DEF_B, fuzzy=True, dayfirst=True)
    except Exception:
        return ""
    # accept only if year, month AND day were all present in the string
    # (i.e. not silently supplied from the defaults) -> rejects partials/relatives
    if a.year == b.year and a.month == b.month and a.day == b.day:
        return a.strftime("%Y-%m-%d")
    return ""


def convert_entities(entities_file, output_dir, normalize_dates=True):
    with open(entities_file, "r", encoding="utf-8") as f:
        entities = list(csv.DictReader(f))

    grouped = {}
    for row in entities:
        etype = row["type"]
        try:
            props = ast.literal_eval(row["properties"])
        except Exception:
            props = {}
        grouped.setdefault(etype, []).append((row["id"], props))

    date_stats = Counter()  # (key -> kept/normalized/blanked) tallies
    for etype, rows in grouped.items():
        all_keys = sorted({k for _, props in rows for k in props})
        date_keys = [k for k in all_keys if k.lower().endswith("date")]
        filepath = os.path.join(output_dir, f"{etype}.csv")
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([":ID"] + all_keys + [":LABEL"])
            for eid, props in rows:
                values = []
                for k in all_keys:
                    v = props.get(k, "")
                    if normalize_dates and k in date_keys:
                        raw = str(v).strip()
                        nv = normalize_date(v)
                        if raw and not nv:
                            date_stats[f"{k}:blanked"] += 1
                        elif raw and nv != raw:
                            date_stats[f"{k}:normalized"] += 1
                        elif raw:
                            date_stats[f"{k}:kept"] += 1
                        v = nv
                    values.append(v)
                writer.writerow([eid] + values + [etype])

    print(f"  {len(grouped)} entity type file(s) written to {output_dir}/")
    if normalize_dates and date_stats:
        print("  date normalization:")
        for key in sorted(date_stats):
            print(f"    {key}: {date_stats[key]}")


def convert_relations(relations_file, output_dir, normalize_dates=True):
    """Property columns are discovered dynamically (mirroring
    convert_entities()) rather than hardcoded to context/source -- a fixed
    list here previously dropped role/date (added to memberOf/affiliation
    relations in prepare_er_input.py) silently, the same class of bug as
    that RELATION_FIELDS omission, just one stage further downstream."""
    with open(relations_file, "r", encoding="utf-8") as f:
        relations = list(csv.DictReader(f))

    reserved = {"start_id", "end_id", "relation"}
    grouped = {}
    for row in relations:
        rel = row["relation"]
        grouped.setdefault(rel, []).append(row)

    date_stats = Counter()
    for rel, rows in grouped.items():
        prop_keys = sorted({k for row in rows for k in row if k not in reserved and row.get(k)})
        date_keys = [k for k in prop_keys if k.lower().endswith("date")]
        filepath = os.path.join(output_dir, f"{rel}.csv")
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([":START_ID", ":END_ID", ":TYPE"] + prop_keys)
            for row in rows:
                values = []
                for k in prop_keys:
                    v = row.get(k, "")
                    if normalize_dates and k in date_keys:
                        raw = str(v).strip()
                        nv = normalize_date(v)
                        if raw and not nv:
                            date_stats[f"{rel}.{k}:blanked"] += 1
                        elif raw and nv != raw:
                            date_stats[f"{rel}.{k}:normalized"] += 1
                        elif raw:
                            date_stats[f"{rel}.{k}:kept"] += 1
                        v = nv
                    values.append(v)
                writer.writerow([row["start_id"], row["end_id"], rel] + values)

    print(f"  {len(grouped)} relation type file(s) written to {output_dir}/")
    if normalize_dates and date_stats:
        print("  relation date normalization:")
        for key in sorted(date_stats):
            print(f"    {key}: {date_stats[key]}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert entities/relations CSVs to Neo4j bulk import format."
    )
    parser.add_argument("--entities",  required=True, help="Cleaned entities CSV")
    parser.add_argument("--relations", required=True, help="Cleaned relations CSV")
    parser.add_argument("--output",    required=True, help="Base output directory")
    parser.add_argument("--no-date-norm", action="store_true",
                        help="Disable ISO date normalization of *Date properties")
    args = parser.parse_args()

    entities_dir  = os.path.join(args.output, "entities")
    relations_dir = os.path.join(args.output, "relations")
    os.makedirs(entities_dir,  exist_ok=True)
    os.makedirs(relations_dir, exist_ok=True)

    print("Converting entities...")
    convert_entities(args.entities, entities_dir, normalize_dates=not args.no_date_norm)

    print("Converting relations...")
    convert_relations(args.relations, relations_dir, normalize_dates=not args.no_date_norm)

    print(f"\nDone. Import files ready in {args.output}/")


if __name__ == "__main__":
    main()
