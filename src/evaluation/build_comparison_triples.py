# -*- coding: utf-8 -*-
"""
Build a model's comparison_triples.csv for Sentence-BERT triple evaluation.

Replicates TripleSelectionforEvaluation.ipynb: resolves id-based relations to
(label, type) via the entities file, attaches the extracted quote as
evidence_sentence, and keeps only the emails present in the golden set. Header
relations drop out automatically (their source has no email-number).

Output columns: email_id, subject_label, predicate, object_label,
                evidence_sentence, subject_type, object_type, source_type

Usage:
    python src/evaluation/build_comparison_triples.py \
        --entities results/extractions/gptoss/final_outputs/entities_final.csv \
        --relations results/extractions/gptoss/final_outputs/relations_final.csv \
        --output   results/extractions/gptoss/evaluation_triples/comparison_triples.csv \
        --source_type GptOss
"""

import os
import re
import json
import argparse
from pathlib import Path

import pandas as pd


def get_entity_label_and_type(row):
    try:
        props = json.loads(row.get("properties", "{}"))
    except Exception:
        props = {}
    t = row.get("type", "Unknown")

    def g(key):
        v = props.get(key)
        return str(v).strip() if v is not None else ""

    if t == "Person":        label = g("personName") or g("personEmail") or "Unknown Person"
    elif t == "Paper":       label = g("paperTitle") or "Unknown Paper"
    elif t == "Dataset":     label = g("datasetName") or "Unknown Dataset"
    elif t == "Method":      label = g("methodName") or "Unknown Method"
    elif t == "Task":        label = g("taskName") or "Unknown Task"
    elif t == "Metric":      label = g("metricName") or "Unknown Metric"
    elif t == "Email":       label = f"Email {g('mailNum')}" if g("mailNum") else "Unknown Email"
    elif t == "MailThread":  label = f"Thread {g('threadID')}" if g("threadID") else "Unknown Thread"
    elif t == "SubmissionID":label = g("identifier") or "Unknown Submission"
    elif t == "Conference":  label = g("confTitle") or "Unknown Conference"
    elif t == "Journal":     label = g("journalTitle") or "Unknown Journal"
    elif t == "PaperStatus":
        s, d = g("status"), g("statusDate")
        label = f"{s} ({d})" if s and d else (s or d or "Unknown Status")
    elif t == "Meeting":     label = g("meetAgenda") or "Unknown Meeting"
    else:                    label = list(props.values())[0] if props else t
    return label, t


def normalize_email_id(source_str):
    if pd.isna(source_str):
        return None
    digits = re.findall(r"\d+", str(source_str))
    return f"e{digits[0]}" if digits else str(source_str).strip()


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--relations", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source_type", required=True, help="model tag, e.g. GptOss")
    ap.add_argument("--golden", default=str(repo / "datasets/extraction_gold/refined_golden_set_target.csv"))
    args = ap.parse_args()

    gold = pd.read_csv(args.golden)
    target_emails = set(gold["email_id"].unique())
    print(f"golden target emails: {len(target_emails)}")

    ent = pd.read_csv(args.entities)
    rel = pd.read_csv(args.relations).dropna(subset=["source"]).copy()
    rel["normalized_source"] = rel["source"].apply(normalize_email_id)

    id_label, id_type = {}, {}
    for _, r in ent.iterrows():
        lab, typ = get_entity_label_and_type(r)
        id_label[r["id"]] = lab
        id_type[r["id"]] = typ

    rf = rel[rel["normalized_source"].isin(target_emails)].copy()
    rows = []
    for _, r in rf.iterrows():
        s_id, o_id = r["start_id"], r["end_id"]
        if s_id in id_label and o_id in id_label:
            rows.append({
                "email_id": r["normalized_source"],
                "subject_label": id_label[s_id],
                "predicate": r["relation"],
                "object_label": id_label[o_id],
                "evidence_sentence": r["context"],
                "subject_type": id_type[s_id],
                "object_type": id_type[o_id],
                "source_type": args.source_type,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"saved {args.output} with {len(df)} triples across "
          f"{df['email_id'].nunique()} emails")


if __name__ == "__main__":
    main()
