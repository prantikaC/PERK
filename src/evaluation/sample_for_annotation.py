# -*- coding: utf-8 -*-
"""
Sample N emails from extracted KG data and generate a triple annotation CSV.
The output can be loaded into the annotation tool for golden-set creation.

Usage:
    python sample_for_annotation.py \
        --entities  entities_final.csv \
        --relations relations_final.csv \
        --emails    PATRA_v3_cleaned_v5.txt \
        --output    golden_set_candidates.csv \
        --n_emails  250 \
        --seed      42
"""

import argparse
import json
import os
import random

import pandas as pd


def get_entity_label(props_str, entity_type):
    try:
        props = json.loads(props_str)
    except Exception:
        return "UNKNOWN"

    def get(key):
        return props.get(key, "").strip()

    if entity_type == "Person":
        return get("personName") or get("personEmail") or "Unknown Person"
    elif entity_type == "Paper":
        return get("paperTitle") or "Unknown Paper"
    elif entity_type == "Dataset":
        return get("datasetName") or "Unknown Dataset"
    elif entity_type == "Method":
        return get("methodName") or "Unknown Method"
    elif entity_type == "Task":
        return get("taskName") or "Unknown Task"
    elif entity_type == "Metric":
        return get("metricName") or "Unknown Metric"
    elif entity_type == "Email":
        num = get("mailNum")
        return f"Email {num}" if num else "Unknown Email"
    elif entity_type == "MailThread":
        tid = get("threadID")
        return f"Thread {tid}" if tid else "Unknown Thread"
    elif entity_type == "SubmissionID":
        return get("identifier") or "Unknown Submission"
    elif entity_type == "Conference":
        return get("confTitle") or "Unknown Conference"
    elif entity_type == "Journal":
        return get("journalTitle") or "Unknown Journal"
    elif entity_type == "PaperStatus":
        status, date = get("status"), get("statusDate")
        if status and date:
            return f"{status} ({date})"
        return status or date or "Unknown Status"
    elif entity_type == "Meeting":
        return get("meetAgenda") or "Unknown Meeting"

    if props:
        return str(list(props.values())[0])
    return entity_type


def load_email_texts(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    raw_emails = [e.strip() for e in content.split("EMAIL_END") if e.strip()]
    return {f"e{i + 1}": text for i, text in enumerate(raw_emails)}


def main():
    parser = argparse.ArgumentParser(
        description="Sample emails and extract triples for annotation."
    )
    parser.add_argument("--entities",  required=True, help="entities_final.csv")
    parser.add_argument("--relations", required=True, help="relations_final.csv")
    parser.add_argument("--emails",    required=True,
                        help="Raw email text file (EMAIL_END delimited)")
    parser.add_argument("--output",    required=True, help="Output CSV for annotation")
    parser.add_argument("--n_emails",  type=int, default=250,
                        help="Number of emails to sample (default: 250)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading data...")
    df_ent = pd.read_csv(args.entities)
    df_rel = pd.read_csv(args.relations)
    email_texts = load_email_texts(args.emails)

    id_to_label = {
        row["id"]: get_entity_label(row["properties"], row["type"])
        for _, row in df_ent.iterrows()
    }

    available_emails = [
        e for e in df_rel["source"].unique()
        if e != "header" and e in email_texts
    ]
    print(f"  Found {len(available_emails)} valid emails.")

    selected = (
        random.sample(available_emails, args.n_emails)
        if len(available_emails) > args.n_emails
        else available_emails
    )
    print(f"  Selected {len(selected)} emails for annotation.")

    triples = []
    for _, row in df_rel.iterrows():
        if row["source"] not in selected:
            continue

        s_id, o_id = row["start_id"], row["end_id"]
        if s_id not in id_to_label or o_id not in id_to_label:
            continue

        context = email_texts.get(row["source"], "")
        context_safe = context.replace('"', "'").replace("\n", " [NEWLINE] ")[:1000]

        evidence = row.get("context", "")
        if pd.isna(evidence):
            evidence = ""
        evidence_safe = str(evidence).replace('"', "'").replace("\n", " ")

        triples.append({
            "subject":            id_to_label[s_id],
            "predicate":          row["relation"],
            "object":             id_to_label[o_id],
            "evidence_sentence":  evidence_safe,
            "full_email_context": context_safe,
            "original_source_id": row["source"],
        })

    pd.DataFrame(triples).to_csv(args.output, index=False)
    print(f"\nCreated '{args.output}' with {len(triples)} triples from {len(selected)} emails.")


if __name__ == "__main__":
    main()
