# -*- coding: utf-8 -*-
"""
Generate answers only: NL question -> Cypher -> Neo4j -> DB output.

This is kg_eval_v4.py's Cypher-generation/self-repair/relaxation-retry
pipeline with the LLM-judge step removed entirely -- it produces
GENERATED_CYPHER and DB_OUTPUT for every question but does NOT decide
pass/fail. Judging is a separate, later step (see deterministic_eval.py),
kept out of this script so answer generation and answer grading are two
independently re-runnable stages: you can re-grade the same DB_OUTPUT column
under different matching strictness without spending any more LLM calls on
Cypher generation.

Reuses the exact same connection, validator, repair, and retry logic as
kg_eval_v4.py by importing it directly -- no duplicated pipeline code, so
fixes made there (Rules 1-15, relaxation retry, Cartesian-product check,
etc.) apply here automatically.

Usage:
    python kg_qa.py --model gpt --ontology ontology/PERKOnto.json --input PRASHNA_PATRA_v5.csv
"""

import argparse
import os

import pandas as pd

from kg_eval_v4 import (
    connect_neo4j, explain_check, load_ontology, QueryStructureValidator,
    _looks_like_refusal, _repair_cypher, build_cypher_chain, _is_null_only_result,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate KG-QA answers only (NL -> Cypher -> Neo4j), no LLM judging."
    )
    parser.add_argument("--model", required=True,
                         help="KG prefix used to look up env vars, e.g. 'gpt' -> GPT_NEO4J_URI")
    parser.add_argument("--ontology", default="PERKOnto.json")
    parser.add_argument("--input", default="ISWC_PRASHNA_PATRA_v3.csv",
                         help="Input QA dataset CSV with NLQ and GOLD_ANS columns")
    parser.add_argument("--max_attempts", type=int, default=4,
                         help="Max self-repair attempts before declaring unanswerable")
    parser.add_argument("--tag", default="",
                         help="Optional label appended to output filenames, e.g. 'run1'")
    parser.add_argument("--outdir", default=".",
                         help="Directory for RESULTS/LOG output files")
    parser.add_argument("--llm_model", default="gpt-4.1",
                         help="NL->Cypher generator model name. For a local vLLM server, this "
                              "is the --served-model-name (or full HF id) the server was started "
                              "with, e.g. 'Qwen/Qwen2.5-32B-Instruct-AWQ'.")
    parser.add_argument("--llm_base_url", default=None,
                         help="OpenAI-compatible endpoint for the NL->Cypher generator "
                              "(e.g. http://localhost:8000/v1 for a local vLLM server). "
                              "Omit to use the real OpenAI API.")
    parser.add_argument("--llm_api_key", default=None,
                         help="API key for --llm_base_url; defaults to 'EMPTY' (vLLM servers "
                              "don't check it). Ignored when --llm_base_url is omitted.")
    return parser.parse_args()


def main():
    args = parse_args()
    prefix = args.model.upper()

    graph, clean_schema = connect_neo4j(prefix)
    print("\n--- CLEANED SCHEMA SEEN BY LLM ---")
    print(clean_schema)
    print("----------------------------------\n")

    ontology_data, true_ontology_str = load_ontology(args.ontology)
    validator = QueryStructureValidator(ontology_data)
    cypher_chain = build_cypher_chain(
        llm_model=args.llm_model, base_url=args.llm_base_url, api_key=args.llm_api_key
    )

    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    tag_part = f"_{args.tag}" if args.tag else ""
    os.makedirs(args.outdir, exist_ok=True)
    output_csv = os.path.join(args.outdir, f"KG_QA_ANSWERS_{prefix}_{input_stem}{tag_part}.csv")
    output_log = os.path.join(args.outdir, f"KG_QA_LOG_{prefix}_{input_stem}{tag_part}.txt")

    df = pd.read_csv(args.input)
    db_outputs, generated_cyphers, failure_tags = [], [], []

    with open(output_log, "w", encoding="utf-8") as log_f:

        def log(text):
            print(text)
            log_f.write(text + "\n")
            log_f.flush()

        log(f"Generating answers for {prefix} KG on {len(df)} questions (no LLM judging)...")

        for index, row in df.iterrows():
            question = row["NLQ"]
            gold_ans = str(row["GOLD_ANS"])
            log(f"\n--- {index + 1}/{len(df)} | Q: {question} ---")
            log(f"Gold Answer: {gold_ans}")

            gen_cypher = ""
            db_output_str = ""
            feedback_history = []
            is_valid = False
            failure_tag = "none"

            for attempt in range(1, args.max_attempts + 1):
                try:
                    combined_feedback = ""
                    if feedback_history:
                        combined_feedback = (
                            "\n\nYOUR PREVIOUS ATTEMPT(S) FAILED VALIDATION FOR THESE REASONS:\n"
                            + "\n".join(f"- {f}" for f in feedback_history)
                            + "\nFix ALL of these issues in your new output."
                        )

                    gen_cypher = cypher_chain.invoke({
                        "schema":        clean_schema,
                        "true_ontology": true_ontology_str,
                        "question":      question,
                        "feedback":      combined_feedback
                    })
                    gen_cypher = gen_cypher.replace("```cypher", "").replace("```", "").strip()

                    if _looks_like_refusal(gen_cypher):
                        is_valid = True
                        failure_tag = "unanswerable_by_schema"
                        if not gen_cypher.strip().upper().startswith("UNANSWERABLE"):
                            gen_cypher = "UNANSWERABLE: " + gen_cypher
                        break

                    gen_cypher, repair_notes = _repair_cypher(gen_cypher, validator)
                    if repair_notes:
                        log(f"  [Attempt {attempt}] Auto-repaired: {'; '.join(repair_notes)}")

                    struct_valid, struct_reason = validator.check_cypher(gen_cypher)
                    if not struct_valid:
                        feedback_history.append(struct_reason)
                        failure_tag = "ontology_or_structure"
                        log(f"  [Attempt {attempt}] Structural check failed: {struct_reason}")
                        continue

                    syntax_valid, syntax_reason = explain_check(graph, gen_cypher)
                    if not syntax_valid:
                        feedback_history.append(f"Cypher syntax error (via EXPLAIN): {syntax_reason}")
                        failure_tag = "syntax_error"
                        log(f"  [Attempt {attempt}] EXPLAIN syntax check failed: {syntax_reason}")
                        continue

                    try:
                        raw = graph.query(gen_cypher)
                    except Exception as exec_e:
                        feedback_history.append(f"Cypher runtime error on execution: {exec_e}")
                        failure_tag = "execution_error"
                        log(f"  [Attempt {attempt}] Execution failed: {exec_e}")
                        continue

                    db_output_str = str(raw)
                    is_valid = True
                    # "empty_result": the query returned literally zero rows.
                    # "negative_or_null_result": rows came back but every value is
                    # False/None -- this is NOT necessarily wrong. It's the correct,
                    # expected shape for a legitimate boolean-False answer, and is
                    # indistinguishable here from an over-constrained query that
                    # matched nothing; DETERMINISTIC_VERDICT (which compares against
                    # GOLD_ANS) is the actual correctness signal, not this tag.
                    if db_output_str in ("[]", ""):
                        failure_tag = "empty_result"
                    elif _is_null_only_result(db_output_str):
                        failure_tag = "negative_or_null_result"
                    else:
                        failure_tag = "none"
                    break

                except Exception as e:
                    feedback_history.append(str(e))
                    failure_tag = "generation_error"
                    log(f"  [Attempt {attempt}] Generation error: {e}")

            if gen_cypher.upper().startswith("UNANSWERABLE"):
                db_output_str = "The information could not be extracted from the KG (no valid schema path)."
            elif not is_valid:
                log(f"  Failed to generate valid Cypher after {args.max_attempts} attempts. Skipping DB query.")
                db_output_str = db_output_str or "The information could not be extracted from the KG."
                gen_cypher = gen_cypher or "Failed to generate valid Cypher."
            else:
                is_negative_result = db_output_str in ("[]", "") or _is_null_only_result(db_output_str)
                if is_negative_result:
                    was_boolean_false = db_output_str not in ("[]", "")
                    relaxed_fb = (
                        ("\n\nYOUR PREVIOUS QUERY RETURNED FALSE (count(x) > 0 was 0):\n"
                         if was_boolean_false else
                         "\n\nYOUR PREVIOUS QUERY RETURNED ZERO ROWS:\n") + gen_cypher +
                        "\nThe answer may exist but the query is likely OVER-CONSTRAINED. Produce ONE "
                        "less restrictive query that stays faithful to the question: drop non-essential "
                        "WHERE filters and any extra MATCH joins that scope to a specific meeting, thread, "
                        "or date not strictly required. In particular, if the WHERE clause ANDs together "
                        "multiple literal phrase-fragments from the question onto the SAME property (e.g. "
                        "requiring a Task name to contain several separate words at once), keep only the "
                        "one or two most distinctive terms -- the graph's actual wording may not restate "
                        "every qualifier verbatim. Keep the SAME core entities and relationship; do "
                        "NOT broaden entity types or invent new joins. "
                        "Do NOT relax by OR-ing in ANY generic, domain-wide word as a stand-in for "
                        "one specific multi-word concept from the question -- this applies even to a "
                        "SINGLE such word, not just three or more. In a single-topic-domain corpus like "
                        "this one (historical NLP research), a word like \"nlp\", \"historical\", "
                        "\"research\", \"interpret\", \"language\", or \"archive\" appears on nearly "
                        "EVERY node of that type -- e.g. OR-ing in just \"nlp\" alone when relaxing a "
                        "Conference query matches almost every conference in the graph, and OR-ing in "
                        "just \"historian\" alongside \"interpretability\" for a dataset query can "
                        "surface a completely unrelated dataset that happens to share one of those two "
                        "words. Adding even one such word does not faithfully relax the query, it changes "
                        "its meaning and will produce a false positive. This is especially dangerous for "
                        "a boolean (count(x) > 0) question that just returned False, or for a question "
                        "whose gold answer may legitimately be 'not available': turning a correct empty/"
                        "False result into a fabricated match via a generic-word OR is WORSE than leaving "
                        "it empty/False, since a wrong 'yes'/fabricated answer is a more serious error "
                        "than correctly reporting no match. If the only available relaxation involves "
                        "adding any generic, domain-wide word as an OR alternative, do not do it -- reply "
                        "exactly 'UNANSWERABLE' instead. "
                        "If the over-constrained filter is a DATE or YEAR -- whether checked via "
                        "date(x).year or via a substring on an identifier/title (e.g. s.identifier "
                        "CONTAINS \"2019\", c.confTitle CONTAINS \"2019\") -- relax the day/month "
                        "precision or an agenda/keyword filter instead, but NEVER drop the year check "
                        "itself. This corpus reuses similar event/venue names across many different "
                        "years (recurring Meetings like \"coordination call\"; the same conference like "
                        "JCDL recurring across 2020, 2022, 2023...), so dropping only the year filter "
                        "will match a completely different year's unrelated event/submission and produce "
                        "a wrong answer that looks superficially correct. "
                        "If no faithful, less-constrained "
                        "query is possible, reply exactly 'UNANSWERABLE'."
                    )
                    try:
                        rc = cypher_chain.invoke({"schema": clean_schema, "true_ontology": true_ontology_str,
                                                  "question": question, "feedback": relaxed_fb})
                        rc = rc.replace("```cypher", "").replace("```", "").strip()
                        rc, _ = _repair_cypher(rc, validator)
                        if _looks_like_refusal(rc):
                            log(f"  [Relaxation] LLM declined to relax (returned UNANSWERABLE-style refusal): {rc}")
                        else:
                            struct_ok, struct_reason = validator.check_cypher(rc)
                            if not struct_ok:
                                log(f"  [Relaxation] relaxed query failed structural check: {struct_reason} | Cypher: {rc}")
                            else:
                                syntax_ok, syntax_reason = explain_check(graph, rc)
                                if not syntax_ok:
                                    log(f"  [Relaxation] relaxed query failed EXPLAIN syntax check: {syntax_reason} | Cypher: {rc}")
                                else:
                                    rraw = graph.query(rc)
                                    rraw_str = str(rraw)
                                    if rraw_str not in ("[]", "") and not _is_null_only_result(rraw_str):
                                        db_output_str = rraw_str
                                        gen_cypher = rc
                                        failure_tag = "none"
                                        log("  [Relaxation] over-constrained query relaxed -> rows found")
                                    else:
                                        log(f"  [Relaxation] relaxed query ran but still returned no rows: {rc}")
                    except Exception as e:
                        log(f"  [Relaxation] skipped ({e})")
                if len(db_output_str) > 3000:
                    log("Warning: Massive DB result detected (kept in full for grading; log preview only).")

            db_outputs.append(db_output_str)
            generated_cyphers.append(gen_cypher)
            failure_tags.append(failure_tag)

            log(f"Cypher: {gen_cypher}")
            log(f"DB Output: {db_output_str[:150]}...")
            log(f"Failure tag: {failure_tag}")

        df["GENERATED_CYPHER"] = generated_cyphers
        df["DB_OUTPUT"]        = db_outputs
        df["FAILURE_TAG"]      = failure_tags
        df.to_csv(output_csv, index=False)

        log(f"\n====================================")
        log(f"{prefix} answer generation complete on {len(df)} questions.")
        log("Failure taxonomy breakdown:")
        for tag, count in pd.Series(failure_tags).value_counts().items():
            log(f"  {tag}: {count}")
        log(f"Answers saved to {output_csv}")
        log(f"Log saved to {output_log}")
        log(f"(No pass/fail judged here -- run deterministic_eval.py on {output_csv} to grade it.)")
        log(f"====================================")


if __name__ == "__main__":
    main()
