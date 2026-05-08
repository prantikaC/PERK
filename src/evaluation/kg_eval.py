# -*- coding: utf-8 -*-
"""
Evaluate PERK QA: NL question -> Cypher -> Neo4j -> LLM utility score.

Usage:
    python kg_eval.py \
        --model   gpt \
        --ontology perk_ontology.json \
        --input   ISWC_PRASHNA_PATRA_v3.csv

    Credentials are read from env vars: <MODEL_UPPER>_NEO4J_URI,
    <MODEL_UPPER>_NEO4J_USERNAME, <MODEL_UPPER>_NEO4J_PASSWORD.
    E.g. --model gpt  =>  GPT_NEO4J_URI, GPT_NEO4J_USERNAME, GPT_NEO4J_PASSWORD
"""

import argparse
import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()


# ==========================================
# 1. CLI
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate QA on a PERK Neo4j KG via NL-to-Cypher."
    )
    parser.add_argument(
        "--model", required=True,
        help="KG prefix used to look up env vars, e.g. 'gpt' -> GPT_NEO4J_URI"
    )
    parser.add_argument("--ontology", default="PERKOnto.json")
    parser.add_argument("--input",    default="ISWC_PRASHNA_PATRA_v3.csv",
                        help="Input QA dataset CSV with NLQ and GOLD_ANS columns")
    return parser.parse_args()


# ==========================================
# 2. Neo4j connection + schema
# ==========================================
def connect_neo4j(prefix):
    uri  = os.getenv(f"{prefix}_NEO4J_URI")
    user = os.getenv(f"{prefix}_NEO4J_USERNAME")
    pwd  = os.getenv(f"{prefix}_NEO4J_PASSWORD")

    if not all([uri, user, pwd]):
        raise ValueError(
            f"Missing credentials for prefix '{prefix}'. "
            f"Set {prefix}_NEO4J_URI, {prefix}_NEO4J_USERNAME, {prefix}_NEO4J_PASSWORD in .env"
        )

    graph = Neo4jGraph(url=uri, username=user, password=pwd)
    graph.refresh_schema()

    raw_schema = graph.schema
    if "The relationships:" in raw_schema:
        clean_schema = raw_schema.split("The relationships:")[0].strip()
    else:
        clean_schema = raw_schema

    return graph, clean_schema


# ==========================================
# 3. Ontology helpers
# ==========================================
def load_ontology(path):
    with open(path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    valid_paths = []
    for rel_name, rel_data in ontology.get("relationships", {}).items():
        rel_type = rel_data.get("type", rel_name)
        for pair in rel_data.get("valid_pairs", []):
            if len(pair) == 2:
                valid_paths.append(f"(:{pair[0]})-[:{rel_type}]->(:{pair[1]})")

    return ontology, "\n".join(valid_paths)


class RegexOntologyChecker:
    def __init__(self, ontology_dict):
        self.valid_nodes = set(ontology_dict.get("nodes", {}).keys())
        self.valid_triplets = set()

        for rel_name, rel_data in ontology_dict.get("relationships", {}).items():
            rel_type = rel_data.get("type", rel_name)
            for pair in rel_data.get("valid_pairs", []):
                if len(pair) == 2:
                    self.valid_triplets.add((pair[0], rel_type, pair[1]))

    def check_cypher(self, cypher_query: str):
        node_pattern = re.compile(
            r'\(\s*([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z0-9_]+)\s*(?:{[^}]+})?\s*\)'
        )
        var_to_label = {}

        for match in node_pattern.finditer(cypher_query):
            var_name, label = match.groups()
            if label not in self.valid_nodes:
                return False, f"Hallucinated Node Label: '{label}' is not in the ontology."
            var_to_label[var_name] = label

        rel_pattern = re.compile(
            r'\(\s*([a-zA-Z0-9_]+)(?:\s*:\s*[a-zA-Z0-9_]+)?\s*\)'
            r'\s*(<-|-)\s*'
            r'\[\s*(?:[a-zA-Z0-9_]+)?\s*:\s*([a-zA-Z0-9_]+).*?\]'
            r'\s*(->|-)\s*'
            r'\(\s*([a-zA-Z0-9_]+)(?:\s*:\s*[a-zA-Z0-9_]+)?\s*\)'
        )

        for match in rel_pattern.finditer(cypher_query):
            source_var, left_arrow, rel_type, right_arrow, target_var = match.groups()

            source_label = var_to_label.get(source_var)
            target_label = var_to_label.get(target_var)

            if not source_label or not target_label:
                return False, (
                    "Used anonymous nodes or undeclared variables in relationship. "
                    "You MUST declare variables and labels (e.g., (p:Person))."
                )

            is_directed_right = right_arrow == "->"
            is_directed_left  = left_arrow  == "<-"
            is_undirected     = right_arrow == "-" and left_arrow == "-"

            valid_forward = (source_label, rel_type, target_label) in self.valid_triplets
            valid_reverse = (target_label, rel_type, source_label) in self.valid_triplets

            if is_directed_right and not valid_forward:
                return False, (
                    f"Invalid Path: ({source_label})-[:{rel_type}]->({target_label}) "
                    "does not exist in ontology."
                )
            if is_directed_left and not valid_reverse:
                return False, (
                    f"Invalid Path: ({source_label})<-[:{rel_type}]-({target_label}) "
                    "does not exist in ontology."
                )
            if is_undirected and not (valid_forward or valid_reverse):
                return False, (
                    f"Invalid Path: ({source_label})-[:{rel_type}]-({target_label}) "
                    "is not a valid pair in any direction."
                )

        return True, "Valid"


# ==========================================
# 4. Text-to-Cypher chain
# ==========================================
CYPHER_SYSTEM = """You are an expert Neo4j Cypher developer.
Convert the user's natural language question into a Cypher query.

LIVE NODE PROPERTIES (Use this strictly for finding property names):
{schema}

TRUE PERK ONTOLOGY RELATIONSHIPS (You MUST ONLY use these exact paths. Ignore any other implied relationships):
{true_ontology}

CRITICAL INSTRUCTIONS:
1. Output ONLY the raw Cypher query. No markdown formatting.
2. NEVER guess property names. You MUST look at the LIVE NODE PROPERTIES block to find the exact string property.
3. NEVER use exact string matching. ALWAYS use case-insensitive substring matching: `WHERE toLower(p.personName) CONTAINS toLower("Sunita")`.
4. If the property you are looking for is not on the node, check if you need to traverse a relationship according to the TRUE PERK ONTOLOGY.
5. DATE HANDLING: ALL date fields (meetDate, taskDate, confDate, statusDate, mailDate) are stored in ISO YYYY-MM-DD format. Use date() for all comparisons.
   a) Year filtering: `WHERE date(m.meetDate).year = 2019`
   b) Month+day filtering: `WHERE date(m.meetDate).month = 5 AND date(m.meetDate).day = 3`
   c) Exact date: `WHERE m.meetDate = "2019-04-08"`
   d) Before/after: `WHERE date(m.meetDate) < date("2019-04-08")`
   e) YYYY-MM partial fields: use CONTAINS, e.g. `WHERE m.confDate CONTAINS "2019-04"`
   f) Guard nulls: `WHERE m.meetDate IS NOT NULL AND date(m.meetDate).year = 2019`
   g) CRITICAL — When the question gives only month+day (e.g. "3rd May", "April 6th") WITHOUT a year, do NOT hardcode a year. Use: `date(m.meetDate).month = 5 AND date(m.meetDate).day = 3`
6. Make sure to include DISTINCT if there is any possibility of duplicate results.
7. YES/NO QUESTIONS: Use the `count() > 0` aggregation to force a TRUE/FALSE output. For availability/attendance questions where a match means TRUE, use `count(m) > 0 AS result`.
8. COMPOUND QUESTIONS (multiple people/entities): Use OR in WHERE — NEVER use UNION. UNION causes variable scope crashes.
   WRONG: MATCH (p1:Person) WHERE ... RETURN p1.name  UNION  RETURN p2.name  ← CRASH
   CORRECT: MATCH (p:Person) WHERE toLower(p.personName) CONTAINS "A" OR toLower(p.personName) CONTAINS "B" RETURN p.affiliation
9. Do NOT query Email nodes — they have no useful properties for answering questions.
10. TASK VS. METRIC/METHOD: If a question describes an active action (e.g., "compiling", "preparing", "working on"), map it to a Task node.
11. MISSING ONTOLOGY PATHS: If no valid path exists between nodes, output: RETURN 'No valid path in schema' AS Error
12. ALWAYS declare variables with their specific labels (e.g., `(p:Person)` instead of `()`) so the ontology checker can parse them.
13. Do NOT use PaperBib. Use Paper, SubmissionID, PaperStatus for paper lifecycle; Journal and Conference for venues.
14. MEETING PROPERTIES: meetAgenda, meetLink, meetTime, meetDate are TEXT PROPERTIES on Meeting nodes — not relationship types.
15. RELATIONSHIP DIRECTIONS (always follow these exactly):
    (Person)-[:worksOn]->(Task)
    (Person)-[:worksWith]->(Method|Dataset)
    (Person)-[:attends]->(Meeting)
    (Paper)-[:hasAuthor]->(Person)
    (SubmissionID)-[:identifies]->(Paper)
    (SubmissionID)-[:movesTo]->(PaperStatus)
    (SubmissionID)-[:inVenue]->(Journal|Conference)
    (Dataset)-[:usedFor]->(Task)
    (Method)-[:usedFor]->(Task)
    (Metric)-[:evaluates]->(Method)
    (Email)-[:partOf]->(MailThread)
16. KEYWORD SEARCH on Task: use broad OR terms rather than all-AND, since task names vary in phrasing.
{feedback}"""


def build_cypher_chain():
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", CYPHER_SYSTEM),
        ("user", "{question}")
    ])
    return prompt | llm | StrOutputParser()


# ==========================================
# 5. Evaluator
# ==========================================
class EvaluationResult(BaseModel):
    is_useful: bool = Field(
        description="True if DB Output fundamentally provides the same correct info as the Gold Answer."
    )
    reason: str = Field(description="Brief explanation of the evaluation.")


def evaluate_utility(eval_llm, question: str, db_output: str, gold_ans: str) -> EvaluationResult:
    gold_empty = "not available" in str(gold_ans).lower() or str(gold_ans).strip() == ""
    db_empty   = (
        db_output in ("[]", "")
        or "not available" in db_output.lower()
        or "could not be extracted" in db_output.lower()
    )

    if gold_empty and db_empty:
        return EvaluationResult(is_useful=True, reason="Both correctly identified missing information.")

    prompt = (
        f"You are evaluating the usability of a Knowledge Graph Question Answering system.\n"
        f"User Question: {question}\n"
        f"Expected Gold Answer: {gold_ans}\n"
        f"Raw Database Output: {db_output}\n"
        f"Does the raw database output contain the correct information to satisfy the user's "
        f"question compared to the Gold Answer?"
    )
    evaluator = eval_llm.with_structured_output(EvaluationResult)
    return evaluator.invoke(prompt)


# ==========================================
# 6. Main loop
# ==========================================
def main():
    args = parse_args()
    prefix = args.model.upper()

    graph, clean_schema = connect_neo4j(prefix)
    print("\n--- CLEANED SCHEMA SEEN BY LLM ---")
    print(clean_schema)
    print("----------------------------------\n")

    ontology_data, true_ontology_str = load_ontology(args.ontology)
    regex_checker = RegexOntologyChecker(ontology_data)

    cypher_chain = build_cypher_chain()
    eval_llm     = ChatOpenAI(model="gpt-5.1", temperature=0)

    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    output_csv = f"KG_UTILITY_RESULTS_{prefix}_{input_stem}.csv"
    output_log = f"KG_UTILITY_LOG_{prefix}_{input_stem}.txt"

    df = pd.read_csv(args.input)

    db_outputs, generated_cyphers, eval_results, eval_reasons = [], [], [], []

    with open(output_log, "w", encoding="utf-8") as log_f:

        def log(text):
            print(text)
            log_f.write(text + "\n")
            log_f.flush()

        log(f"Evaluating {prefix} KG on {len(df)} questions using Regex Checker...")

        for index, row in df.iterrows():
            question = row["NLQ"]
            gold_ans = str(row["GOLD_ANS"])
            log(f"\n--- {index + 1}/{len(df)} | Q: {question} ---")

            gen_cypher = ""
            db_output_str = ""
            feedback = ""
            is_valid = False

            for attempt in range(1, 4):
                try:
                    gen_cypher = cypher_chain.invoke({
                        "schema":        clean_schema,
                        "true_ontology": true_ontology_str,
                        "question":      question,
                        "feedback": (
                            f"\n\nYOUR PREVIOUS CYPHER FAILED ONTOLOGY VALIDATION BECAUSE: {feedback}"
                            "\nFix this violation in your new output."
                        ) if feedback else ""
                    })
                    gen_cypher = gen_cypher.replace("```cypher", "").replace("```", "").strip()

                    is_valid, check_reason = regex_checker.check_cypher(gen_cypher)
                    if is_valid:
                        break
                    else:
                        feedback = check_reason
                        log(f"  [Attempt {attempt}] Regex check failed: {feedback}")

                except Exception as e:
                    feedback = str(e)
                    log(f"  [Attempt {attempt}] Generation error: {e}")

            if not is_valid:
                log("  Failed to generate valid Cypher after 3 attempts. Skipping DB query.")
                db_output_str = "The information could not be extracted from the KG."
                gen_cypher = gen_cypher or "Failed to generate valid Cypher."
            else:
                try:
                    raw = graph.query(gen_cypher)
                    db_output_str = str(raw)
                    if len(db_output_str) > 3000:
                        log("Warning: Massive DB result detected. Truncating for evaluator...")
                        db_output_str = db_output_str[:3000] + "... [TRUNCATED DUE TO SIZE]"
                except Exception as e:
                    log(f"Error executing Cypher: {e}")
                    db_output_str = "Error during execution."

            try:
                result = evaluate_utility(eval_llm, question, db_output_str, gold_ans)
            except Exception as e:
                log(f"Evaluator API Error: {e}")
                result = EvaluationResult(is_useful=False, reason="API Error: evaluation failed.")

            db_outputs.append(db_output_str)
            generated_cyphers.append(gen_cypher)
            eval_results.append(result.is_useful)
            eval_reasons.append(result.reason)

            log(f"Cypher: {gen_cypher}")
            log(f"DB Output: {db_output_str[:100]}...")
            log(f"Eval: {'Pass' if result.is_useful else 'Fail'} ({result.reason})")

            time.sleep(1)

        df["GENERATED_CYPHER"] = generated_cyphers
        df["DB_OUTPUT"]        = db_outputs
        df["IS_USEFUL"]        = eval_results
        df["EVAL_REASON"]      = eval_reasons
        df.to_csv(output_csv, index=False)

        accuracy = sum(eval_results) / len(eval_results) * 100
        log(f"\n====================================")
        log(f"{prefix} Pipeline Evaluation Complete! Accuracy: {accuracy:.2f}%")
        log(f"Data saved to {output_csv}")
        log(f"Logs saved to {output_log}")
        log(f"====================================")


if __name__ == "__main__":
    main()
