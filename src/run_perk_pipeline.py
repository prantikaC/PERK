# -*- coding: utf-8 -*-
"""
End-to-end orchestrator: given a corpus of a user's own emails (in PATRA
format), build a queryable PERK knowledge graph by running every pipeline
stage in this repo, in order, with no manual steps in between.

Stages run, in order (each is also independently runnable -- see README.md
for the per-stage commands and what each script does):

  1. extraction/kg_extraction_pipeline.py
     Header/signature regex extraction + LLM body extraction, merged per
     corpus. Targeted Rules 1-3 (Appendix B.1) all apply here, before entity
     resolution, including the hasAuthor evidence check.
  2. entity_resolution/faiss_blocking.py       -- candidate blocking
  3. entity_resolution/llm_judgement.py        -- MATCH/NO_MATCH judging
  4. entity_resolution/node_fusion.py          -- transitive graph fusion
  5. entity_resolution/normalize_dates.py      -- date normalization on fused entities
  6. entity_resolution/acronym_cluster_audit.py + apply_acronym_merges.py
                                                -- second pass for proper-noun
                                                   entities (conferences,
                                                   journals, organizations)
  7. neo4j/rebuild_paperstatus_chains_pre_import.py
                                                -- clean per-submission
                                                   PaperStatus chains, before
                                                   graph construction
  8. neo4j/clean_kg.py                         -- ontology validation
  9. neo4j/prepare_import.py                   -- per-type CSV split
                                                   (also normalizes dates by
                                                   default)
 10. neo4j/build_perk.py                       -- Neo4j graph load

Input format: a single .txt file of PATRA-delimited emails --

    Thread ID: <id>
    Mail ID: <id>
    Date: <DD Month YYYY>
    From: <name> [<email>]
    To: <name> [<email>], ...
    CC: <name> [<email>], ...        (optional)
    Subject: <subject>

    <body>

    <signature>
    EMAIL_END

Prerequisites (not run by this script):
  - .env with your LLM backend credentials (OPENAI_API_KEY, etc.) and
    Neo4j credentials (NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD, or
    pass --neo4j-uri/--neo4j-user/--neo4j-password).
  - A few-shot golden-set CSV for the LLM entity-resolution judge
    (--golden_set) -- this repo's own datasets/extraction_gold/ is NOT
    the right shape for this; you need entity-pair MATCH/NO_MATCH examples
    (see entity_resolution/generate_golden_candidates.py to build one from
    your own corpus, or reuse this project's if your domain is similar).

Usage:
    python run_perk_pipeline.py \
        --input_file  my_emails.txt \
        --output_dir  runs/my_run \
        --ontology    ontology/PERKOnto.json \
        --golden_set  my_er_golden_set.csv \
        --model openai --model_path gpt-5.1

    # Skip the Neo4j load (just produce neo4j_import/ CSVs to inspect first):
    python run_perk_pipeline.py ... --skip-neo4j

    # Resume a partially-completed run (skips stages whose output already exists):
    python run_perk_pipeline.py ... --resume
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_stage(name, cmd, resume, sentinel_path):
    """Run one pipeline stage as a subprocess. If --resume was passed and
    sentinel_path already exists, skip it. Exits the whole pipeline on
    failure rather than continuing with stale/partial downstream input."""
    print(f"\n{'=' * 70}\nSTAGE: {name}\n{'=' * 70}")
    if resume and sentinel_path and os.path.exists(sentinel_path):
        print(f"[resume] {sentinel_path} already exists -- skipping.")
        return
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nSTAGE '{name}' FAILED (exit code {result.returncode}). Stopping pipeline.")
        sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser(
        description="Build a PERK knowledge graph end-to-end from a user's own emails.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_file", required=True, help="PATRA-format corpus .txt file")
    p.add_argument("--output_dir", required=True, help="Root output directory for this run")
    p.add_argument("--ontology", default=os.path.join(HERE, "..", "ontology", "PERKOnto.json"))
    p.add_argument("--golden_set", required=True,
                    help="Entity-pair MATCH/NO_MATCH golden set CSV, for the ER few-shot prompt")

    # Extraction (Stage 1)
    p.add_argument("--model", required=True,
                    help="Extraction backend: openai, gptoss, llama, gemma, qwen, qwen32b")
    p.add_argument("--model_path", default=None,
                    help="OpenAI model name or HF model id; optional for aliases with a default")
    p.add_argument("--extraction_base_url", default=None,
                    help="OpenAI-compatible endpoint for a local vLLM server (extraction stage)")
    p.add_argument("--gpu", default="0", help="GPU id(s) for local vLLM stages (extraction, ER)")

    # Entity resolution (Stages 2-6)
    p.add_argument("--faiss_threshold", type=float, default=0.6,
                    help="FAISS auto-reject floor (default: 0.6, the value used in the paper)")
    p.add_argument("--er_backend", choices=["vllm", "openai"], default="vllm",
                    help="LLM backend for entity-resolution judging + acronym audit")
    p.add_argument("--er_model", default="Qwen/Qwen2.5-32B-Instruct",
                    help="Model for entity-resolution judging (vLLM model id or OpenAI model name)")
    p.add_argument("--er_base_url", default=None,
                    help="OpenAI-compatible endpoint for a local vLLM server (ER stage)")
    p.add_argument("--llm_confidence_floor", type=float, default=0.0,
                    help="node_fusion.py's --llm_confidence_floor (0.0 unless your judge model's "
                         "confidence scores are meaningfully calibrated -- see node_fusion.py)")
    p.add_argument("--skip_acronym_pass", action="store_true",
                    help="Skip the second-pass proper-noun (conference/journal/org) merge round")

    # Graph construction (Stages 7-10)
    p.add_argument("--neo4j_uri", default=os.getenv("NEO4J_URI"))
    p.add_argument("--neo4j_user", default=os.getenv("NEO4J_USERNAME"))
    p.add_argument("--neo4j_password", default=os.getenv("NEO4J_PASSWORD"))
    p.add_argument("--skip_neo4j", action="store_true",
                    help="Stop after producing neo4j_import/ CSVs; don't load into Neo4j")

    p.add_argument("--resume", action="store_true",
                    help="Skip any stage whose output file already exists")
    args = p.parse_args()

    py = sys.executable
    out = args.output_dir
    entity_res_dir = os.path.join(out, "entity_resolution")
    graph_dir = os.path.join(out, "graph")
    for d in (out, entity_res_dir, graph_dir):
        os.makedirs(d, exist_ok=True)

    extraction_dir = os.path.join(out, "extraction")
    final_entities = os.path.join(extraction_dir, "final_outputs", "entities_final.csv")
    final_relations = os.path.join(extraction_dir, "final_outputs", "relations_final.csv")

    # ---- Stage 1: extraction ----
    cmd = [py, os.path.join(HERE, "extraction", "kg_extraction_pipeline.py"),
           "--model", args.model, "--input_file", args.input_file,
           "--output_dir", extraction_dir]
    if args.model_path:
        cmd += ["--model_path", args.model_path]
    if args.extraction_base_url:
        cmd += ["--base_url", args.extraction_base_url]
    if args.gpu:
        cmd += ["--gpu", args.gpu]
    run_stage("1. Extraction", cmd, args.resume, final_relations)

    # ---- Stage 2: FAISS blocking ----
    grey_zone = os.path.join(entity_res_dir, "grey_zone.csv")
    cmd = [py, os.path.join(HERE, "entity_resolution", "faiss_blocking.py"),
           "--entities", final_entities, "--relations", final_relations,
           "--output", grey_zone, "--threshold", str(args.faiss_threshold),
           "--gpu", args.gpu, "--corpus", args.input_file,
           "--log", os.path.join(entity_res_dir, "step1_faiss.log")]
    run_stage("2. FAISS blocking", cmd, args.resume, grey_zone)

    # ---- Stage 3: LLM judgement ----
    llm_resolved = os.path.join(entity_res_dir, "llm_resolved.csv")
    cmd = [py, os.path.join(HERE, "entity_resolution", "llm_judgement.py"),
           "--grey_zone", grey_zone, "--golden_set", args.golden_set,
           "--output", llm_resolved, "--backend", args.er_backend,
           "--model", args.er_model, "--gpu", args.gpu,
           "--raw_entities", final_entities, "--raw_relations", final_relations,
           "--log", os.path.join(entity_res_dir, "step2_llm_judgement.log")]
    if args.er_base_url:
        cmd += ["--base_url", args.er_base_url]
    run_stage("3. LLM judgement", cmd, args.resume, llm_resolved)

    # ---- Stage 4: node fusion ----
    fused_entities = os.path.join(entity_res_dir, "fused_entities.csv")
    fused_relations = os.path.join(entity_res_dir, "fused_relations.csv")
    cmd = [py, os.path.join(HERE, "entity_resolution", "node_fusion.py"),
           "--llm_resolved", llm_resolved,
           "--raw_entities", final_entities, "--raw_relations", final_relations,
           "--fused_entities", fused_entities, "--fused_relations", fused_relations,
           "--llm_confidence_floor", str(args.llm_confidence_floor),
           "--log", os.path.join(entity_res_dir, "step3_node_fusion.log")]
    run_stage("4. Node fusion", cmd, args.resume, fused_relations)

    # ---- Stage 5: date normalization on fused entities ----
    fused_entities_normdates = os.path.join(entity_res_dir, "fused_entities_normdates.csv")
    cmd = [py, os.path.join(HERE, "entity_resolution", "normalize_dates.py"),
           "--entities", fused_entities, "--relations", fused_relations,
           "--output", fused_entities_normdates]
    run_stage("5. Date normalization", cmd, args.resume, fused_entities_normdates)

    entities_for_next = fused_entities_normdates
    relations_for_next = fused_relations

    # ---- Stage 6: second-pass proper-noun (acronym) merge ----
    if not args.skip_acronym_pass:
        candidate_pairs = os.path.join(entity_res_dir, "acronym_candidate_pairs.csv")
        cmd = [py, os.path.join(HERE, "entity_resolution", "acronym_cluster_audit.py"),
               "--fused_entities", entities_for_next, "--fused_relations", relations_for_next,
               "--output", candidate_pairs, "--backend", args.er_backend,
               "--gpu", args.gpu, "--log", os.path.join(entity_res_dir, "acronym_audit.log")]
        if args.er_base_url:
            cmd += ["--base_url", args.er_base_url]
        run_stage("6a. Acronym cluster audit", cmd, args.resume, candidate_pairs)

        acr_entities = os.path.join(entity_res_dir, "entities_acronymmerged.csv")
        acr_relations = os.path.join(entity_res_dir, "relations_acronymmerged.csv")
        cmd = [py, os.path.join(HERE, "entity_resolution", "apply_acronym_merges.py"),
               "--fused_entities", entities_for_next, "--fused_relations", relations_for_next,
               "--candidate_pairs", candidate_pairs,
               "--output_entities", acr_entities, "--output_relations", acr_relations,
               "--apply", "--log", os.path.join(entity_res_dir, "apply_acronym_merges.log")]
        run_stage("6b. Apply acronym merges", cmd, args.resume, acr_relations)
        entities_for_next, relations_for_next = acr_entities, acr_relations

    # ---- Stage 7: rebuild PaperStatus chains, before graph construction ----
    ps_entities = os.path.join(graph_dir, "entities_psfixed.csv")
    ps_relations = os.path.join(graph_dir, "relations_psfixed.csv")
    cmd = [py, os.path.join(HERE, "neo4j", "rebuild_paperstatus_chains_pre_import.py"),
           "--raw-dir", extraction_dir,
           "--entities_in", entities_for_next, "--relations_in", relations_for_next,
           "--entities_out", ps_entities, "--relations_out", ps_relations, "--apply"]
    run_stage("7. Rebuild PaperStatus chains", cmd, args.resume, ps_relations)

    # ---- Stage 8: ontology validation ----
    clean_entities = os.path.join(graph_dir, "entities_clean.csv")
    clean_relations = os.path.join(graph_dir, "relations_clean.csv")
    cmd = [py, os.path.join(HERE, "neo4j", "clean_kg.py"),
           "--entities_in", ps_entities, "--relations_in", ps_relations,
           "--ontology", args.ontology,
           "--entities_out", clean_entities, "--relations_out", clean_relations,
           "--log_file", os.path.join(graph_dir, "clean_kg.log")]
    run_stage("8. Ontology validation (clean_kg.py)", cmd, args.resume, clean_relations)

    # ---- Stage 9: per-type CSV split (also normalizes dates by default) ----
    import_dir = os.path.join(out, "neo4j_import")
    cmd = [py, os.path.join(HERE, "neo4j", "prepare_import.py"),
           "--entities", clean_entities, "--relations", clean_relations,
           "--output", import_dir]
    run_stage("9. Prepare Neo4j import (prepare_import.py)", cmd, args.resume,
              os.path.join(import_dir, "entities"))

    if args.skip_neo4j:
        print(f"\n--skip_neo4j set. Per-type CSVs are in {import_dir}/. Done.")
        return

    # ---- Stage 10: load into Neo4j ----
    if not (args.neo4j_uri and args.neo4j_user and args.neo4j_password):
        print("\nNo Neo4j credentials given (via --neo4j_uri/--neo4j_user/--neo4j_password "
              "or NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD env vars). "
              f"Per-type CSVs are in {import_dir}/ -- run neo4j/build_perk.py yourself when ready.")
        return
    cmd = [py, os.path.join(HERE, "neo4j", "build_perk.py"),
           "--data_dir", import_dir, "--ontology", args.ontology,
           "--uri", args.neo4j_uri, "--user", args.neo4j_user, "--password", args.neo4j_password]
    run_stage("10. Build Neo4j graph", cmd, args.resume, None)

    print(f"\n{'=' * 70}\nDONE. PERK graph built from {args.input_file}.\n{'=' * 70}")


if __name__ == "__main__":
    main()
