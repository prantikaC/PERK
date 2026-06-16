# -*- coding: utf-8 -*-
"""
Stanford OpenIE baseline runner for PERK (Angeli et al., 2015).

Open Information Extraction is the classic *training-free, schema-free* triple
extractor: it produces (subject, relation, object) tuples from any text using a
dependency parse + learned clause classifier -- no LLM, no labelled data, no
ontology. It is the canonical unsupervised baseline and directly rebuts "you
only skipped the supervised methods": OpenIE needs no in-domain supervision yet,
unlike PERK, it is not ontology-guided, so its relations/types are open and
free-form.

This runs Stanford OpenIE per email (the comparable unit to the gold standard)
and writes triples in the gold-standard column layout
(refined_golden_set_target.csv) so the output can be scored with your existing
triple-level evaluation. OpenIE gives no entity typing, so subject_type /
object_type are empty; the source sentence is stored as evidence_sentence when
the wrapper provides it.

Install & run
-------------
    pip install stanford-openie            # Java 8+ required (you have Java 21)
    # first run auto-downloads CoreNLP (~0.5 GB) and starts a local server
    python openie_extraction.py --limit 25   # smoke test
    python openie_extraction.py               # all 1006 emails
"""

import re
import csv
import argparse
from pathlib import Path

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


def report(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    preds = [r["predicate"] for r in rows]
    distinct = sorted(set(preds))
    in_rel = sum(1 for p in preds if p in PERK_RELATIONS)
    print("\n================ OPENIE BASELINE SUMMARY ================")
    print(f"file: {path}")
    print(f"  triples extracted     : {len(rows)}")
    print(f"  distinct predicates   : {len(distinct)}  (open, surface-form phrases)")
    if preds:
        print(f"  predicates in PERKOnto: {in_rel}/{len(preds)} "
              f"({100*in_rel/len(preds):.1f}%)  (open schema -> expected ~0%)")
    print("Score this CSV against refined_golden_set_target.csv with a lenient")
    print("predicate mapping, since OpenIE predicates are free-form verb phrases.")
    print("=========================================================\n")


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patra", default=str(repo / "datasets/PATRA/PATRA.txt"))
    ap.add_argument("--out", default=str(repo / "results/baselines/openie.csv"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    try:
        from openie import StanfordOpenIE
    except ImportError:
        raise SystemExit("stanford-openie not installed.  Run:  pip install stanford-openie")

    emails = split_emails(Path(args.patra).read_text(encoding="utf-8"))
    if args.limit:
        emails = emails[: args.limit]
    print(f"OpenIE baseline: {len(emails)} emails")

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

    # affinity cap is the wrapper's documented default tuning knob.
    props = {"openie.affinity_probability_cap": 2 / 3}
    n_triples = 0
    with StanfordOpenIE(properties=props) as client:
        for idx, block in enumerate(emails, start=1):
            eid = mail_id_of(block, f"email{idx}")
            if eid in done:
                continue
            try:
                triples = client.annotate(block)
            except Exception as err:                       # noqa: BLE001
                print(f"  {eid}: ERROR {err}")
                continue
            for t in triples:
                s = str(t.get("subject", "")).strip()
                rel = str(t.get("relation", "")).strip()
                o = str(t.get("object", "")).strip()
                if s and rel and o:
                    writer.writerow([eid, s, rel, o, "", "", ""])
                    n_triples += 1
            fh.flush()
            if idx % 25 == 0:
                print(f"  processed {idx}/{len(emails)} emails, {n_triples} new triples")
    fh.close()
    print(f"done -> {out_path} ({n_triples} new triples)")
    report(out_path)


if __name__ == "__main__":
    main()
