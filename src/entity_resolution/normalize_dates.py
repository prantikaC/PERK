# -*- coding: utf-8 -*-
"""
Normalize date fields in extracted entity CSVs to ISO YYYY-MM-DD.
Year is inferred from the source email's mailDate via the relations file,
giving context-aware normalization instead of a fixed default year.

Usage:
    python normalize_dates.py \
        --entities entities_final.csv \
        --relations relations_final.csv \
        --output entities_final_normdates.csv
"""

import argparse
import json
import re
import pandas as pd
from datetime import datetime
from collections import defaultdict

DATE_FIELDS = {"meetDate", "confDate", "statusDate", "taskDate", "mailDate"}

MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

BASELINE_YEAR = 2019


def normalize_date(raw: str, default_year: int) -> str:
    """Convert any date string to YYYY-MM-DD. Returns original if unparseable."""
    if not raw or str(raw).lower() in ("nan", "none", "", "null"):
        return raw
    raw = str(raw).strip()

    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw

    m = re.match(r"^(\d{4})-(\d{1,2})$", raw)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-01"

    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", raw)
    if m:
        d, mo, y = m.groups()
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", raw.lower())
    cleaned = re.sub(
        r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)[,\s]+", "", cleaned
    )

    # If the string itself contains a 4-digit year, prefer it over default_year
    m_yr = re.search(r"\b((?:19|20)\d{2})\b", cleaned)
    explicit_year = int(m_yr.group(1)) if m_yr else None

    month_pat = "|".join(MONTH_MAP.keys())
    m = re.search(r"(?:(\d{1,2})\s+)?(" + month_pat + r")(?:\s+(\d{1,2}))?", cleaned)
    if m:
        day_str, month_str, trailing_day = m.groups()
        month = MONTH_MAP[month_str]
        day = int(day_str) if day_str else (int(trailing_day) if trailing_day else 1)
        year = explicit_year if explicit_year else default_year
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            pass

    m = re.match(r"^(\d{4})$", raw.strip())
    if m:
        return f"{m.group(1)}-01-01"

    return raw


def build_year_map(df_entities: pd.DataFrame, df_relations: pd.DataFrame) -> dict:
    """
    Returns {entity_id: year} using a 3-level fallback:
      1. Year from source email's mailDate (traced via relations)
      2. Year from another ISO date field on the same entity
      3. BASELINE_YEAR (2019)
    """
    entity_props = {}
    for _, row in df_entities.iterrows():
        eid = str(row["id"]).strip()
        try:
            entity_props[eid] = json.loads(row["properties"])
        except Exception:
            entity_props[eid] = {}

    # email entity id → year extracted from mailDate
    email_year = {}
    for eid, props in entity_props.items():
        mail_date = str(props.get("mailDate", ""))
        m = re.search(r"\b((?:19|20)\d{2})\b", mail_date)
        if m:
            email_year[eid] = int(m.group(1))

    # entity id → set of source email ids (from relations source column)
    entity_to_emails = defaultdict(set)
    for _, row in df_relations.iterrows():
        source = str(row.get("source", "")).strip()
        if not source or source.lower() in ("nan", "unknown", "header", ""):
            continue
        entity_to_emails[str(row["start_id"]).strip()].add(source)
        entity_to_emails[str(row["end_id"]).strip()].add(source)

    year_map = {}
    for eid, props in entity_props.items():
        # 1. Source email year
        for email_id in entity_to_emails.get(eid, []):
            if email_id in email_year:
                year_map[eid] = email_year[email_id]
                break
        if eid in year_map:
            continue

        # 2. ISO date on the entity itself
        for field in ("statusDate", "mailDate", "confDate"):
            val = str(props.get(field, ""))
            m = re.match(r"^(\d{4})-", val)
            if m:
                year_map[eid] = int(m.group(1))
                break

        # 3. Baseline
        if eid not in year_map:
            year_map[eid] = BASELINE_YEAR

    return year_map


def normalize_entities(df_entities: pd.DataFrame, year_map: dict) -> tuple[pd.DataFrame, int]:
    normalized_props = []
    changed = 0
    for _, row in df_entities.iterrows():
        eid = str(row["id"]).strip()
        original = str(row["properties"])
        try:
            props = json.loads(original)
        except (json.JSONDecodeError, TypeError):
            normalized_props.append(original)
            continue

        default_year = year_map.get(eid, BASELINE_YEAR)
        updated = False
        for field in DATE_FIELDS:
            if field in props and props[field]:
                norm = normalize_date(str(props[field]), default_year)
                if norm != str(props[field]):
                    props[field] = norm
                    updated = True

        if updated:
            changed += 1
            normalized_props.append(json.dumps(props))
        else:
            normalized_props.append(original)

    df_out = df_entities.copy()
    df_out["properties"] = normalized_props
    return df_out, changed


def main():
    parser = argparse.ArgumentParser(
        description="Normalize date fields in entity CSV using source-email year inference."
    )
    parser.add_argument("--entities",  required=True, help="Input entities CSV (id, type, properties)")
    parser.add_argument("--relations", required=True, help="Relations CSV (needed for year inference)")
    parser.add_argument("--output",    required=True, help="Output normalized entities CSV")
    args = parser.parse_args()

    df_ent = pd.read_csv(args.entities)
    df_rel = pd.read_csv(args.relations)
    print(f"Loaded {len(df_ent)} entities, {len(df_rel)} relations.")

    year_map = build_year_map(df_ent, df_rel)
    df_out, changed = normalize_entities(df_ent, year_map)
    df_out.to_csv(args.output, index=False)

    print(f"Normalized {changed} entities with date fields.")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
