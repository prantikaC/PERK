# -*- coding: utf-8 -*-
"""
Convert per-email entity/relation CSVs into human-readable triple CSVs
for downstream triple evaluation.

Expects per-email files named entities_emailN.csv and relations_emailN.csv
inside each input directory. Writes triples_emailN.csv alongside them.

Usage:
    python convert_to_triples.py --input_dirs /path/to/model1 /path/to/model2
"""

import argparse
import glob
import json
import os

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


def process_folder(folder_path):
    print(f"Processing: {folder_path}")
    entity_files = glob.glob(os.path.join(folder_path, "entities_email*.csv"))

    if not entity_files:
        print(f"  No entity files found.")
        return

    count = 0
    for ent_file in entity_files:
        try:
            filename  = os.path.basename(ent_file)
            email_num = filename.replace("entities_email", "").replace(".csv", "")
            rel_file  = os.path.join(folder_path, f"relations_email{email_num}.csv")

            if not os.path.exists(rel_file):
                continue

            df_ent = pd.read_csv(ent_file)
            df_rel = pd.read_csv(rel_file)

            if df_rel.empty:
                continue

            id_to_label = {
                row["id"]: get_entity_label(row["properties"], row["type"])
                for _, row in df_ent.iterrows()
            }

            triples = []
            for _, row in df_rel.iterrows():
                s_id, o_id = row["start_id"], row["end_id"]
                if s_id in id_to_label and o_id in id_to_label:
                    triples.append({
                        "subject":   id_to_label[s_id],
                        "predicate": row["relation"],
                        "object":    id_to_label[o_id],
                    })

            if triples:
                out_path = os.path.join(folder_path, f"triples_email{email_num}.csv")
                pd.DataFrame(triples).to_csv(out_path, index=False)
                count += 1

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    print(f"  -> Generated triples for {count} emails.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert per-email entity/relation CSVs to triple CSVs."
    )
    parser.add_argument(
        "--input_dirs", nargs="+", required=True,
        help="One or more directories containing entities_emailN.csv files"
    )
    args = parser.parse_args()

    for folder in args.input_dirs:
        if os.path.exists(folder):
            process_folder(folder)
        else:
            print(f"Folder not found: {folder}")

    print("\nConversion complete.")


if __name__ == "__main__":
    main()
