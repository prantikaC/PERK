# -*- coding: utf-8 -*-
"""
Single-command orchestrator for the full PERK extraction pipeline:

    header/signature parsing (header_signature_parser.py)
      -> body LLM extraction, WITH header context (kg_extraction_pipeline.py)
      -> venue affiliation (header_signature_parser.py, deferred pass)
      -> reconcile body Persons against header owners (filter_body_entities.py)
      -> merge header + body into ER-ready entities_final.csv/relations_final.csv
         (prepare_er_input.py)

This exists so nobody has to remember to run four separate scripts in the
right order, or realize that kg_extraction_pipeline.py's own built-in header
parsing is the OLD, cruder one and header_signature_parser.py is the real,
later replacement (Team/Organization/EmailID modeling, signature-derived
role/affiliation) that should be feeding the body-extraction LLM's context
instead.

Ordering: header/signature entities (Email, MailThread, EmailID, Person,
Team, Organization, and every header/signature-derived relation EXCEPT
Journal/Conference affiliation) are resolved for the WHOLE corpus first,
before any LLM call -- so the body-extraction prompt for every email
already has real, richer Person/Team ids to reuse instead of building its
own redundant, cruder ones. The ONE piece that cannot run first is
Journal/Conference affiliation-via-signature, because it needs to know
which Journal/Conference the body LLM found in that same email -- that
runs as a short second header-side pass after body extraction completes,
using header_signature_parser.py's own apply_venue_affiliation().

This script imports header_signature_parser.py and kg_extraction_pipeline.py
directly (so the header-context hand-off can happen in-process, per email)
but calls filter_body_entities.py and prepare_er_input.py as subprocesses
(their job is a plain CSV-in/CSV-out transform once extraction is done --
no reason to duplicate or import their internals). None of the four
scripts' own tested logic is modified beyond header_signature_parser.py and
kg_extraction_pipeline.py each gaining a small, backward-compatible hook
for this hand-off; run standalone (their own CLI, unchanged), they behave
exactly as before.

Usage:
    python run_full_extraction_pipeline.py \
        --model openai --model_path gpt-5.1 \
        --input_file  ../../data/PATRA/MyPATRA.txt \
        --output_dir  ../../data/myPERK \
        --prefix      mypatra_openai
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
ER_DIR = os.path.join(SRC_DIR, "entity_resolution")
DEFAULT_ONTOLOGY = os.path.join(os.path.dirname(SRC_DIR), "ontology", "PERKOnto.json")

sys.path.insert(0, THIS_DIR)
import header_signature_parser as hsp  # noqa: E402
import kg_extraction_pipeline as kgp   # noqa: E402


def run_subprocess(cmd, tag):
    print(f"\n[{tag}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"[{tag}] Command failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Full PERK extraction pipeline, single command")
    parser.add_argument("--model", required=True,
                        choices=["gemma", "llama", "qwen", "qwen32b", "openai", "gptoss"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True, help="Root output directory (subfolders created inside)")
    parser.add_argument("--prefix", required=True, help="Output file prefix for the final ER-ready CSVs")
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY)
    parser.add_argument("--prompt_file", default=str(kgp.DEFAULT_PROMPT))
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--save_debug", action="store_true", default=True)
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = kgp.DEFAULT_MODEL_PATHS.get(args.model.lower())
    if not args.model_path:
        parser.error(f"--model_path is required for model '{args.model}'")

    header_dir = os.path.join(args.output_dir, "header_extraction")
    body_dir = os.path.join(args.output_dir, "body_extraction")
    filtered_dir = os.path.join(args.output_dir, "body_extraction_filtered")
    er_input_dir = os.path.join(args.output_dir, "entity_resolution")
    for d in [args.output_dir, header_dir, body_dir, filtered_dir, er_input_dir]:
        os.makedirs(d, exist_ok=True)

    # kg_extraction_pipeline.process_email()/merge_and_report() expect these
    # derived paths on the args object it's given -- build a Namespace that
    # matches what its own main() sets up, pointed at body_dir.
    kgp_args = argparse.Namespace(
        model=args.model, model_path=args.model_path, base_url=args.base_url,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_new_tokens=args.max_new_tokens,
        save_debug=args.save_debug, output_dir=body_dir,
    )
    kgp_args.entity_dir = os.path.join(body_dir, "entity_extractions")
    kgp_args.relation_dir = os.path.join(body_dir, "relation_extractions")
    kgp_args.final_dir = os.path.join(body_dir, "final_outputs")
    kgp_args.final_entities = os.path.join(kgp_args.final_dir, "entities_final.csv")
    kgp_args.final_relations = os.path.join(kgp_args.final_dir, "relations_final.csv")
    for d in [kgp_args.entity_dir, kgp_args.relation_dir, kgp_args.final_dir]:
        os.makedirs(d, exist_ok=True)

    logger = kgp.setup_logger(os.path.join(kgp_args.final_dir, "extraction_process.log"))
    logger.info("=" * 80)
    logger.info(f" FULL PIPELINE | Model: {args.model} | Path: {args.model_path}")
    logger.info("=" * 80)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        all_emails = [e.strip() for e in f.read().split('EMAIL_END') if e.strip()]
    logger.info(f"Found {len(all_emails)} emails.")

    # ---- Stage 1: header/signature pass (whole corpus, before any LLM call) ----
    logger.info("Stage 1/4: header/signature parsing...")
    hsp.reset_state()
    hsp.collect_best_names(all_emails)
    pending_by_num = {}  # email_num (1-indexed, corpus order) -> pending dict
    for i, email_text in enumerate(all_emails):
        pending = hsp.process_email_header(email_text, email_num=i + 1)
        if pending is not None:
            pending_by_num[i + 1] = pending
    logger.info(f"Header pass resolved {len(pending_by_num)}/{len(all_emails)} emails "
                f"(others had no Mail ID). Entities so far: {len(hsp.entities)}")

    # Pre-seed kg_extraction_pipeline's own id counters past whatever header
    # already claimed, for every entity type/prefix the two modules share
    # (currently just Person/"pn"). Without this, each module's own counter
    # starts independently at 1, so a genuinely NEW body-only entity (e.g.
    # a person mentioned by name only once, never in a header) can land on
    # the exact same id string as an existing header entity purely by
    # coincidence -- confirmed on a real-inbox test corpus: body-side "Person A"
    # and header-side "Person B" both independently minted
    # "pn3". prepare_er_input.py's renumbering step can no longer safely
    # fix this after the fact once body relations may also legitimately
    # contain a literal header id (via the HDR_ context-reuse mechanism
    # above) -- it can't tell "this pn3 means the coincidental body
    # collision" from "this pn3 correctly means the header entity", so it
    # remapped BOTH, corrupting the correct references. Preventing the
    # collision at the source avoids the ambiguity entirely.
    shared_types = set(kgp.PREFIX_MAP) & set(hsp.PREFIX_MAP)
    for etype in shared_types:
        kgp.id_counters[etype] = max(kgp.id_counters[etype], hsp.id_counters.get(etype, 0))
    logger.info(f"Pre-seeded body id counters past header's max for shared types: "
                f"{ {t: kgp.id_counters[t] for t in shared_types} }")

    # ---- Stage 2: body LLM extraction, using header context ----
    logger.info("Stage 2/4: body LLM extraction (using header/signature context)...")
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()
    engine, sampling, client = kgp.load_model(kgp_args)

    # Built directly here instead of via header_signature_parser.load_venue_
    # cross_reference(), which anchors mailNum on Email-type rows in body's
    # OWN entities_final.csv -- body extraction never mints those anymore
    # under this architecture (header owns Email entities exclusively; see
    # header_context_override in kg_extraction_pipeline.process_email), so
    # that lookup was always empty and apply_venue_affiliation() below never
    # fired for anyone, corpus-wide. Confirmed on a real-inbox test corpus:
    # a journal Editor-in-Chief and an Associate Editor were both
    # correctly extracted as Person entities but ended up with zero
    # relations at all. Reading each email's own freshly-written per-email
    # file right after processing it ties venues to mail_num directly and
    # correctly, with no dependency on Email rows existing at all.
    mailnum_to_venues = {}

    start_time = time.time()
    for i, email_text in enumerate(all_emails):
        email_num = i + 1
        pending = pending_by_num.get(email_num)
        if pending is not None:
            header_context, context_id_map = hsp.build_llm_context(
                pending["participants"], pending["org_email_hints"],
                pending["mail_date"], pending["subject"])
            current_email_id = pending["own_email_id"]
            extra_valid_ids = {eid for eid, _etype, _props in pending["participants"]}
            org_email_hints = pending["org_email_hints"]
        else:
            header_context = "No persons in headers."
            context_id_map = {}
            current_email_id = "unknown"
            extra_valid_ids = set()
            org_email_hints = []

        try:
            kgp.process_email(
                email_num, email_text, engine, sampling, client, system_prompt, kgp_args, logger,
                header_context_override=header_context,
                current_email_id_override=current_email_id,
                extra_valid_ids=extra_valid_ids,
                org_email_hints_override=org_email_hints,
                context_id_map=context_id_map,
            )
        except Exception as e:
            logger.error(f"CRITICAL ERROR on Email {email_num}: {e}")
            logger.exception("Full traceback:")
            continue

        if pending is not None:
            per_email_entities_path = os.path.join(
                kgp_args.entity_dir, f"entities_email{email_num}.csv")
            try:
                with open(per_email_entities_path, newline='', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        if row['type'] in ('Journal', 'Conference'):
                            try:
                                props = json.loads(row['properties'])
                            except (json.JSONDecodeError, TypeError):
                                props = {}
                            label_field = 'journalTitle' if row['type'] == 'Journal' else 'confTitle'
                            mailnum_to_venues.setdefault(pending["mail_num"], []).append(
                                (row['id'], row['type'], props.get(label_field, "")))
            except FileNotFoundError:
                pass

        if email_num % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Progress: {email_num}/{len(all_emails)} | Avg: {elapsed / email_num:.2f}s/email")

    kgp.merge_and_report(kgp_args, logger)

    # ---- Stage 3: venue affiliation (deferred header-side pass) ----
    logger.info("Stage 3/4: venue affiliation (Journal/Conference found in body, "
                "linked back to signature-named individuals)...")
    for pending in pending_by_num.values():
        venues = mailnum_to_venues.get(pending["mail_num"], [])
        hsp.apply_venue_affiliation(pending, venues)
    hsp.write_outputs(header_dir)
    header_entities_csv = os.path.join(header_dir, "header_entities.csv")
    header_relations_csv = os.path.join(header_dir, "header_relations.csv")

    # ---- Stage 4: reconcile + merge into ER-ready CSVs (subprocess: pure CSV transforms) ----
    logger.info("Stage 4/4: reconciling body Persons against header owners, merging into "
                "ER-ready entities_final.csv/relations_final.csv...")
    run_subprocess([
        sys.executable, os.path.join(THIS_DIR, "filter_body_entities.py"),
        "--entities_csv", kgp_args.final_entities,
        "--relations_csv", kgp_args.final_relations,
        "--header_entities", header_entities_csv,
        "--header_relations", header_relations_csv,
        "--output_dir", filtered_dir,
    ], "filter_body_entities")

    run_subprocess([
        sys.executable, os.path.join(ER_DIR, "prepare_er_input.py"),
        "--header_entities", header_entities_csv,
        "--header_relations", header_relations_csv,
        "--body_entities", os.path.join(filtered_dir, "body_entities.csv"),
        "--body_relations", os.path.join(filtered_dir, "body_relations.csv"),
        "--ontology", args.ontology,
        "--output_dir", er_input_dir,
        "--prefix", args.prefix,
    ], "prepare_er_input")

    logger.info("=" * 80)
    logger.info("FULL PIPELINE COMPLETE.")
    logger.info(f"Final entities/relations: {er_input_dir}/{args.prefix}_entities_final.csv, "
                f"{args.prefix}_relations_final.csv")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
