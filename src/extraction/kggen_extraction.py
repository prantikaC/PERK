# -*- coding: utf-8 -*-
"""
KGGen baseline runner for PERK (Mo et al., 2025, arXiv:2502.09956; pip install kg-gen).

KGGen is a zero-shot, training-free, *schema-free* text-to-KG extractor: an LLM
emits open (subject, predicate, object) triples which are then aggregated and
LLM-clustered. It uses no ontology, so its predicates and node types are open
vocabulary -- this is the schema-free counterpart to PERK's ontology-guided
extraction, and a fair same-assumption baseline (unlike supervised REBEL/DeepKE).

This runs KGGen per email (the comparable unit to the gold standard) with the
SAME LLM you use for PERK (GPT-5.1) and writes triples in the gold-standard
column layout (refined_golden_set_target.csv) so the output can be scored with
your existing triple-level evaluation. KGGen gives no per-triple typing or
evidence sentence, so subject_type/object_type/evidence_sentence are left empty.

Install & run
-------------
    pip install kg-gen
    export OPENAI_API_KEY=sk-...
    python kggen_extraction.py --limit 25          # smoke test
    python kggen_extraction.py                      # all 1006 emails
    python kggen_extraction.py --model openai/gpt-4o   # KGGen's paper default

Note: for KGGen's *native* corpus-level behaviour (aggregate + cluster across all
sources into one global graph) run the library directly on the concatenated
corpus; that output is not keyed per email and so is scored via KGQA / MINE-style
recall rather than the per-email gold set.
"""

import os
import re
import csv
import time
import argparse
from pathlib import Path

# PERKOnto inventory, for the on-schema drift report.
PERK_TYPES = {
    "Person", "Paper", "Conference", "Journal", "SubmissionID", "PaperStatus",
    "Meeting", "Dataset", "Method", "Task", "Metric",
}
PERK_RELATIONS = {
    "hasAuthor", "identifies", "inVenue", "movesTo", "worksWith",
    "worksOn", "usedFor", "evaluates", "uses", "attends",
}
GOLD_COLUMNS = [
    "email_id", "subject_label", "predicate", "object_label",
    "evidence_sentence", "subject_type", "object_type",
]


def split_emails(text):
    return [b.strip() for b in text.split("EMAIL_END") if b.strip()]


def mail_id_of(block, fallback):
    m = re.search(r"Mail ID:\s*([^\n\r]+)", block)
    return m.group(1).strip() if m else fallback


def triples_of(graph):
    """Return a list of (subject, predicate, object) from a KGGen Graph, robustly
    across kg-gen versions (relations as 3-tuples, lists, or dicts)."""
    rels = getattr(graph, "relations", None)
    if rels is None and isinstance(graph, dict):
        rels = graph.get("relations")
    out = []
    for r in (rels or []):
        if isinstance(r, (tuple, list)) and len(r) >= 3:
            out.append((str(r[0]), str(r[1]), str(r[2])))
        elif isinstance(r, dict):
            s = r.get("subject") or r.get("start") or r.get("source")
            p = r.get("predicate") or r.get("relation") or r.get("edge")
            o = r.get("object") or r.get("end") or r.get("target")
            if s and p and o:
                out.append((str(s), str(p), str(o)))
    return out


def report(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    preds = [r["predicate"] for r in rows]
    distinct = sorted(set(preds))
    in_rel = sum(1 for p in preds if p in PERK_RELATIONS)
    print("\n================ KGGEN BASELINE SUMMARY ================")
    print(f"file: {path}")
    print(f"  triples extracted     : {len(rows)}")
    print(f"  distinct predicates   : {len(distinct)}")
    if preds:
        print(f"  predicates in PERKOnto: {in_rel}/{len(preds)} "
              f"({100*in_rel/len(preds):.1f}%)  "
              f"(open schema -> expected to be low)")
    print("Score this CSV against refined_golden_set_target.csv (lenient")
    print("predicate mapping needed, since KGGen predicates are open vocabulary).")
    print("========================================================\n")


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patra", default=str(repo / "datasets/PATRA/PATRA.txt"))
    ap.add_argument("--out", default=str(repo / "results/baselines/kggen.csv"))
    ap.add_argument("--model", default="openai/gpt-5.1",
                    help="litellm model id (e.g. openai/gpt-5.1, openai/gpt-4o)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="must be 1.0 for the gpt-5 family; use 0.0 for gpt-4o")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cluster", action="store_true",
                    help="enable KGGen per-call entity/edge clustering")
    args = ap.parse_args()

    try:
        from kg_gen import KGGen
    except ImportError:
        raise SystemExit("kg-gen not installed.  Run:  pip install kg-gen")

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set OPENAI_API_KEY in the environment.")

    kg = KGGen(model=args.model, temperature=args.temperature, api_key=key)

    emails = split_emails(Path(args.patra).read_text(encoding="utf-8"))
    if args.limit:
        emails = emails[: args.limit]
    print(f"KGGen baseline: {len(emails)} emails, model={args.model}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add(row["email_id"])

    new_file = not out_path.exists()
    fh = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(GOLD_COLUMNS)

    n_triples = 0
    for idx, block in enumerate(emails, start=1):
        eid = mail_id_of(block, f"email{idx}")
        if eid in done:
            continue
        try:
            graph = kg.generate(input_data=block, cluster=args.cluster)
        except TypeError:
            graph = kg.generate(input_data=block)        # older signature
        except Exception as err:                          # noqa: BLE001
            print(f"  {eid}: ERROR {err}")
            time.sleep(2)
            continue
        for s, p, o in triples_of(graph):
            writer.writerow([eid, s, p, o, "", "", ""])
            n_triples += 1
        fh.flush()
        if idx % 25 == 0:
            print(f"  processed {idx}/{len(emails)} emails, {n_triples} new triples")
    fh.close()
    print(f"done -> {out_path} ({n_triples} new triples)")
    report(out_path)


if __name__ == "__main__":
    main()
