# -*- coding: utf-8 -*-
"""
Evaluate PERK QA: NL question -> Cypher -> Neo4j -> LLM utility score.

Improvements over the original version:
  1. Cypher syntax pre-check via Neo4j EXPLAIN, before ontology/regex checks.
     Catches malformed arrow syntax (e.g. "-[:REL]<-") deterministically,
     without wasting an execution attempt or misclassifying it downstream.
  2. Connectivity check: detects disconnected MATCH clauses (a common cause
     of silent Cartesian-product bugs, e.g. "WITH m" followed by a second
     MATCH that never references m via an edge).
  3. Boolean-precedence check: flags ambiguous "A AND B OR C" patterns in
     WHERE clauses (Cypher, like SQL, binds AND tighter than OR, which
     silently produces wrong results rather than an error).
  4. Ontology conformance check (your original RegexOntologyChecker logic),
     now applied per-MATCH-clause so it doesn't miss violations buried in a
     later clause.
  5. Cumulative repair feedback: all issues found across attempts are shown
     to the LLM, not just the latest one, so earlier fixes aren't undone.
  6. Failure taxonomy: every row is tagged with WHERE it failed (syntax,
     connectivity, precedence, ontology, execution, empty_result,
     negative_or_null_result, or none),
     so you can see which layer of the pipeline is driving errors.

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
import ast
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv(override=True)


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
    parser.add_argument("--max_attempts", type=int, default=4,
                        help="Max self-repair attempts before declaring unanswerable")
    parser.add_argument("--tag", default="",
                        help="Optional label appended to output filenames, e.g. 'run1' or 'scoped'")
    parser.add_argument("--outdir", default=".",
                        help="Directory for RESULTS/LOG output files")
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


def explain_check(graph, cypher_query: str):
    """
    Runs Neo4j's EXPLAIN on the query. This validates syntax and planability
    WITHOUT executing the query or touching data. Catches malformed arrow
    syntax, unbound variables, invalid clause ordering, etc. -- errors that
    a regex-based ontology checker cannot reliably anticipate.
    Returns (is_valid, error_message).
    """
    try:
        graph.query(f"EXPLAIN {cypher_query}")
        return True, ""
    except Exception as e:
        return False, str(e)


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


class QueryStructureValidator:
    """
    Combines four independent checks that were previously conflated or
    missing:
      - ontology conformance (labels, relation types, directions)
      - MATCH-clause connectivity (catches Cartesian-product bugs)
      - boolean precedence in WHERE clauses (catches silent AND/OR bugs)
    Syntax validity is checked separately via explain_check(), since it
    requires a live Neo4j connection rather than static text analysis.

    Kept as `RegexOntologyChecker` alias below for backward compatibility
    with any external code/paper text referencing the old name.
    """

    def __init__(self, ontology_dict):
        self.valid_nodes = set(ontology_dict.get("nodes", {}).keys())
        self.valid_triplets = set()
        self.valid_relation_types = set()
        # Maps node label -> set of valid property names, e.g.
        # {"SubmissionID": {"identifier"}, "Conference": {"confTitle", "confDate", "confVenue"}}
        self.node_properties = {
            label: set(data.get("properties", []))
            for label, data in ontology_dict.get("nodes", {}).items()
        }

        for rel_name, rel_data in ontology_dict.get("relationships", {}).items():
            rel_type = rel_data.get("type", rel_name)
            self.valid_relation_types.add(rel_type)
            for pair in rel_data.get("valid_pairs", []):
                if len(pair) == 2:
                    self.valid_triplets.add((pair[0], rel_type, pair[1]))

    # ---------- Check 0: malformed arrow syntax (cheap, instant, no DB call) ----------
    _MALFORMED_ARROW_PATTERNS = [
        (re.compile(r'\]\s*<-'), "']<-' (closing bracket immediately followed by '<-'). "
                                  "A reversed-direction relationship must be written as "
                                  "'<-[:REL]-', with the '<-' BEFORE the bracket, not after it."),
        (re.compile(r'->\s*\['), "'->[' used to OPEN a relationship (arrow before bracket on the "
                                  "left side). Forward relationships are written '-[:REL]->', with "
                                  "the arrow only on the right side."),
        (re.compile(r'-\s*<-'), "'-<-' (a plain dash immediately followed by another arrow). "
                                 "Use a single, correctly placed arrow: '<-[:REL]-' or '-[:REL]->'."),
    ]

    def check_arrow_syntax(self, cypher_query: str):
        """
        Deterministic, regex-based lint for the single most common malformed
        relationship pattern observed in practice: mixing a plain dash with
        an arrow on the same side, e.g. "-[:hasAuthor]<-" instead of the
        correct "<-[:hasAuthor]-". This is checked BEFORE the ontology-path
        regex, because that regex requires a well-formed arrow to recognize
        an edge at all -- a malformed arrow like this one is invisible to
        it and would otherwise pass through undetected until a live Neo4j
        EXPLAIN call catches it (slower, and only catches it after an LLM
        generation round-trip).
        """
        issues = []
        for pattern, message in self._MALFORMED_ARROW_PATTERNS:
            if pattern.search(cypher_query):
                issues.append(f"Malformed relationship arrow syntax: {message}")
        return issues

    # ---------- Shared helper: map variable names to their declared labels ----------
    _NODE_DECL_PATTERN = re.compile(
        r'\(\s*([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z0-9_]+)\s*(?:{[^}]+})?\s*\)'
    )

    def _extract_var_to_label(self, cypher_query: str):
        var_to_label = {}
        for match in self._NODE_DECL_PATTERN.finditer(cypher_query):
            var_name, label = match.groups()
            var_to_label[var_name] = label
        return var_to_label

    # ---------- Check 1: ontology conformance ----------
    def check_ontology(self, cypher_query: str):
        node_pattern = self._NODE_DECL_PATTERN
        var_to_label = {}
        issues = []

        for match in node_pattern.finditer(cypher_query):
            var_name, label = match.groups()
            if label not in self.valid_nodes:
                issues.append(f"Hallucinated node label: '{label}' is not in the ontology.")
            else:
                var_to_label[var_name] = label

        rel_pattern = re.compile(
            r'\(\s*(?:([a-zA-Z0-9_]+))?(?:\s*:\s*([a-zA-Z0-9_]+))?\s*\)'
            r'\s*(<-|-)\s*'
            r'\[\s*(?:[a-zA-Z0-9_]+)?\s*:\s*([a-zA-Z0-9_]+)[^\]]*\]'
            r'\s*(->|-)\s*'
            r'(?=\(\s*(?:([a-zA-Z0-9_]+))?(?:\s*:\s*([a-zA-Z0-9_]+))?\s*\))'
        )
        # NOTE: the trailing node is matched via a lookahead (not consumed).
        # This is essential for chained multi-hop patterns like
        # (p)-[:R1]->(m)-[:R2]->(d): without the lookahead, finditer's
        # non-overlapping matches would consume "(m:Method)" as part of the
        # first edge, leaving nothing for the second edge to anchor on, so
        # :R2 would silently never be checked. With the lookahead, each
        # match ends right before the shared node, so the next finditer
        # call can re-anchor on it and pick up the following edge.
        #
        # Both node positions capture (var, inline_label) separately, and
        # the relationship-type bracket uses [^\]]* (not .*?) so it can
        # NEVER cross a ']'. Both fixes matter together: with the old
        # mandatory-variable node pattern, an anonymous node like (:Team)
        # couldn't satisfy either position, which forced .*? to backtrack
        # PAST the first ']' looking for a later one where the (var-only)
        # target lookahead would succeed -- silently swallowing an entire
        # intermediate hop and misattributing its target to the wrong
        # relationship type (confirmed: (p)-[:memberOf]->(:Team)-[:affiliation]
        # ->(o:Organization) was reported as the nonexistent direct edge
        # "(Person)-[:memberOf]->(Organization)"). Anonymous nodes are
        # completely normal, valid Cypher and must resolve via their own
        # inline label now, not just via a variable declared elsewhere.

        found_any_edge = False
        for match in rel_pattern.finditer(cypher_query):
            found_any_edge = True
            (source_var, source_label_inline, left_arrow, rel_type,
             right_arrow, target_var, target_label_inline) = match.groups()

            if rel_type not in self.valid_relation_types:
                issues.append(
                    f"Hallucinated relationship type: '{rel_type}' is not in the ontology. "
                    f"Valid types are: {sorted(self.valid_relation_types)}."
                )
                continue

            source_label = source_label_inline or var_to_label.get(source_var)
            target_label = target_label_inline or var_to_label.get(target_var)

            if not source_label or not target_label:
                issues.append(
                    "Used an undeclared variable in a relationship (not an inline-labeled "
                    "anonymous node, and not declared with a label anywhere else). Every node "
                    "in a relationship pattern must resolve to a label, e.g. (p:Person) or (:Team)."
                )
                continue

            is_directed_right = right_arrow == "->"
            is_directed_left  = left_arrow  == "<-"
            is_undirected     = right_arrow == "-" and left_arrow == "-"

            valid_forward = (source_label, rel_type, target_label) in self.valid_triplets
            valid_reverse = (target_label, rel_type, source_label) in self.valid_triplets

            if is_directed_right and not valid_forward:
                issues.append(
                    f"Invalid path: ({source_label})-[:{rel_type}]->({target_label}) "
                    "does not exist in the ontology."
                )
            if is_directed_left and not valid_reverse:
                issues.append(
                    f"Invalid path: ({source_label})<-[:{rel_type}]-({target_label}) "
                    "does not exist in the ontology."
                )
            if is_undirected and not (valid_forward or valid_reverse):
                issues.append(
                    f"Invalid path: ({source_label})-[:{rel_type}]-({target_label}) "
                    "is not a valid pair in either direction in the ontology."
                )

        return issues

    # ---------- Check 1b: property-name conformance ----------
    # A handful of empty-result failures (e.g. querying a property that
    # sounds plausible but was never defined on that node type) look
    # identical to a data-sparsity issue from the outside -- both return
    # empty results with no error. This check catches the subset that are
    # actually a property-name mismatch, using the same var_to_label
    # mapping as check_ontology.
    _DOT_ACCESS_PATTERN = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b')
    # Cypher built-in functions/pseudo-properties that are never real node
    # properties, to avoid false positives (e.g. "count(t)" is not "t.count").
    _RESERVED_NON_PROPERTIES = {"id", "labels", "type", "properties", "elementId"}

    def check_property_names(self, cypher_query: str):
        issues = []
        var_to_label = self._extract_var_to_label(cypher_query)

        for var, prop in self._DOT_ACCESS_PATTERN.findall(cypher_query):
            label = var_to_label.get(var)
            if label is None:
                continue  # variable not a labeled node we can check (e.g. a WITH alias)
            if prop in self._RESERVED_NON_PROPERTIES:
                continue
            valid_props = self.node_properties.get(label)
            if valid_props is not None and prop not in valid_props:
                issues.append(
                    f"Hallucinated property: '{var}.{prop}' -- '{prop}' is not a defined "
                    f"property of {label}. Valid properties for {label} are: "
                    f"{sorted(valid_props)}."
                )
        return issues

    # ---------- Check 2: variable scope tracking ----------
    @staticmethod
    def _mask_string_literals(text):
        """
        Replace the CONTENTS of quoted string literals with underscores
        (keeping the quote chars and overall length) so downstream regex
        heuristics don't mistake text inside a string -- e.g. a regex
        literal like '(?i).*review.*' -- for real Cypher syntax. Without
        this, the substring "review." inside that literal looks exactly
        like a variable dot-access ("review.something"), causing a false
        "Variable not in scope" flag that no query rewrite can ever fix,
        since the literal itself is legitimate and necessary.
        """
        return re.sub(
            r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"",
            lambda m: m.group(0)[0] + "_" * (len(m.group(0)) - 2) + m.group(0)[0],
            text,
        )

    def check_variable_scope(self, cypher_query: str):
        """
        Tracks which variables are in scope as the query progresses through
        MATCH/WITH/WHERE/RETURN clauses, and flags references to variables
        that have gone out of scope. This catches a distinct bug from
        check_connectivity: a WITH clause explicitly narrows scope to only
        the variables it lists (optionally renamed via 'AS'), dropping any
        variable not included. A later clause referencing a dropped
        variable (e.g. "WHERE p.id = ..." after "WITH t" without p) is a
        genuine Neo4j semantic error ("Variable p not defined"), not a
        connectivity/Cartesian-product issue.

        This is a heuristic over clause text, not a full Cypher parser, but
        covers the common patterns seen in generated queries. It runs
        alongside (not instead of) the EXPLAIN-based syntax pre-check,
        since it is free (no DB round-trip) and gives an immediate,
        specific repair message.
        """
        cypher_query = self._mask_string_literals(cypher_query)
        issues = []
        node_var_pattern = re.compile(r'\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\b')
        rel_var_pattern = re.compile(r'\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:')
        dot_access_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.')

        clause_pattern = re.compile(
            r'\b(OPTIONAL\s+MATCH|MATCH|WITH|WHERE|RETURN)\b', re.IGNORECASE
        )
        parts = clause_pattern.split(cypher_query)

        scope = set()
        i = 1
        while i < len(parts) - 1:
            keyword = parts[i].strip().upper()
            text = parts[i + 1]

            if keyword in ("MATCH", "OPTIONAL MATCH"):
                scope |= set(node_var_pattern.findall(text))
                scope |= set(rel_var_pattern.findall(text))

            elif keyword == "WITH":
                new_scope = set()
                # Split on top-level commas (good enough heuristic; nested
                # commas inside function calls are rare in these queries).
                for item in text.split(","):
                    item = item.strip()
                    if not item:
                        continue
                    # "WITH DISTINCT a, b" only prefixes DISTINCT onto the
                    # first item once it's comma-split (e.g. "DISTINCT a",
                    # "b") -- strip it so "DISTINCT m" is recognized as the
                    # bare variable 'm' instead of an unparseable expression
                    # that silently falls through and drops 'm' from scope.
                    item = re.sub(r'^DISTINCT\s+', '', item, flags=re.IGNORECASE)
                    as_match = re.search(r'\bAS\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', item, re.IGNORECASE)
                    if as_match:
                        new_scope.add(as_match.group(1))
                    elif re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', item):
                        new_scope.add(item)
                    # Complex expressions without AS (rare) are skipped
                    # rather than guessed, to avoid false positives.
                scope = new_scope

            elif keyword in ("WHERE", "RETURN"):
                used_vars = set(dot_access_pattern.findall(text))
                for v in used_vars:
                    if v not in scope:
                        issues.append(
                            f"Variable '{v}' is referenced in {keyword} but is not in scope at "
                            f"that point in the query. This usually means an earlier WITH clause "
                            f"dropped it -- every variable needed later must be explicitly listed "
                            f"in the WITH clause (e.g. 'WITH {v}, t' instead of 'WITH t')."
                        )

            i += 2

        return issues

    # ---------- Check 3: connectivity across MATCH clauses ----------
    def check_connectivity(self, cypher_query: str):
        """
        Heuristic check for disconnected MATCH clauses. A common bug pattern:
            MATCH (m:Meeting) WHERE ... WITH m
            MATCH (p:Person)-[:worksOn]->(t:Task) WHERE ...
            RETURN m.someProperty
        Here `m` is carried via WITH but never appears inside a relationship
        pattern after the WITH -- the two MATCH clauses form an unintended
        Cartesian product rather than a genuine multi-hop traversal.

        This is a heuristic, not a full Cypher parser: it flags cases where
        a WITH-carried variable is never referenced inside any relationship
        pattern `(...)-[...]-(...)`  in a subsequent MATCH clause, while the
        query's RETURN clause depends on that variable together with
        entities introduced in the later MATCH. For full rigor, consider
        integrating a proper Cypher parser (e.g. libcypher-parser bindings).
        """
        _CYPHER_KEYWORDS = {
            "DISTINCT", "AS", "ORDER", "BY", "ASC", "DESC", "LIMIT", "SKIP",
            "WHERE", "AND", "OR", "NOT", "NULL", "TRUE", "FALSE", "COUNT",
            "COLLECT", "SUM", "AVG", "MIN", "MAX", "WITH", "MATCH", "RETURN",
        }
        issues = []
        with_vars = set()
        for m in re.finditer(r'\bWITH\s+([^\n]+)', cypher_query, re.IGNORECASE):
            vars_in_with = re.findall(r'\b([a-zA-Z0-9_]+)\b', m.group(1))
            with_vars.update(v for v in vars_in_with if v.upper() not in _CYPHER_KEYWORDS)

        if not with_vars:
            return issues  # single-MATCH queries or no WITH -> nothing to check

        # A WITH-carried variable can only be "disconnected" from something
        # if there's an actual MATCH/OPTIONAL MATCH clause AFTER the WITH to
        # be disconnected from. Counting MATCH occurrences across the WHOLE
        # query (the previous approach) over-counts: OPTIONAL MATCH clauses
        # entirely BEFORE the WITH also contain the substring "MATCH", so a
        # query with multiple pre-WITH OPTIONAL MATCHes and nothing at all
        # after WITH was incorrectly treated as having a post-WITH clause to
        # check -- with zero real post-WITH edges, the connectivity check
        # below always fails trivially, flagging a false disconnection on
        # every such query (confirmed: two OPTIONAL MATCHes before a WITH
        # that only feeds a WHERE+RETURN, nothing after WITH at all).
        first_with_match_early = re.search(r'\bWITH\b', cypher_query, re.IGNORECASE)
        if not first_with_match_early or not re.search(
            r'\bMATCH\b', cypher_query[first_with_match_early.end():], re.IGNORECASE
        ):
            return issues

        edge_pattern = re.compile(
            r'\(\s*([a-zA-Z0-9_]+)[^()]*\)\s*(?:<-|-)\s*\[[^\]]*\]\s*(?:->|-)\s*\(\s*([a-zA-Z0-9_]+)'
        )

        # Locate the first WITH keyword as a standalone word -- a plain
        # substring search (e.g. .find("WITH")) would false-positive match
        # inside relation names like "worksWith", splitting the query in
        # the wrong place.
        first_with_match = re.search(r'\bWITH\b', cypher_query, re.IGNORECASE)
        with_idx = first_with_match.start() if first_with_match else 0

        # Variables that appear inside a relationship pattern in clauses
        # AFTER the first WITH.
        post_with_text = cypher_query[with_idx:]
        vars_in_edges_post_with = set()
        for m in edge_pattern.finditer(post_with_text):
            vars_in_edges_post_with.update(m.groups())

        # A WITH-carried variable is NOT actually disconnected if it was
        # already bound through a real relationship pattern in a PRE-WITH
        # MATCH clause together with another variable that continues on
        # into a post-WITH edge -- that shared variable IS the connection,
        # even though the carried var itself is only being passed through
        # unchanged for the final RETURN. e.g.:
        #   MATCH (p:Person)-[:worksWith]->(m:Method) WHERE ...
        #   WITH DISTINCT p, m
        #   MATCH (p)-[:worksOn]->(t:Task) WHERE ...
        #   RETURN m.methodName
        # `m` never reappears in an edge after WITH, but it was bound
        # alongside `p` (which does), so this is a genuine connected
        # traversal, not a Cartesian product. Group variables that co-occur
        # in the same pre-WITH MATCH clause's edge patterns.
        pre_with_text = cypher_query[:with_idx]
        pre_with_clauses = re.split(r'\bMATCH\b', pre_with_text, flags=re.IGNORECASE)[1:]
        pre_with_groups = []
        for clause in pre_with_clauses:
            vars_in_clause = set()
            for m in edge_pattern.finditer(clause):
                vars_in_clause.update(m.groups())
            if vars_in_clause:
                pre_with_groups.append(vars_in_clause)

        # Simplify: any WITH-carried var that is used in RETURN but never
        # appears in an edge pattern after WITH, AND has no pre-WITH
        # group-mate that continues into a post-WITH edge, is a likely
        # disconnection.
        return_text = " ".join(re.findall(r'RETURN[^\n]*', cypher_query, re.IGNORECASE))
        for var in with_vars:
            used_in_return = re.search(rf'\b{re.escape(var)}\b\.', return_text) or \
                              re.search(rf'\b{re.escape(var)}\b', return_text)
            if not used_in_return or var in vars_in_edges_post_with:
                continue
            transitively_connected = any(
                var in grp and (grp & vars_in_edges_post_with) for grp in pre_with_groups
            )
            if transitively_connected:
                continue
            issues.append(
                    f"Possible disconnected query: variable '{var}' is carried via WITH and used "
                    f"in RETURN, but never appears inside a relationship pattern in a later MATCH "
                    f"clause. This likely produces an unintended Cartesian product rather than a "
                    f"real multi-hop traversal. Ensure '{var}' is connected to the other matched "
                    f"entities via an explicit relationship pattern, e.g. "
                    f"(m)-[:REL]->(otherVar), or restructure as a single connected MATCH pattern."
                )
        return issues

    # ---------- Check 3: boolean precedence in WHERE clauses ----------
    @staticmethod
    def _strip_function_call_parens(text: str) -> str:
        """
        Remove parens that belong to ordinary function calls (toLower(...),
        date(...), etc.), leaving only parens that are genuine boolean
        grouping. Repeatedly strips the innermost "word(...)" pattern so
        nested calls (e.g. date(toLower(x))) are fully removed in a few
        passes; any "(" ")" left afterwards must be real grouping parens,
        since they were never immediately preceded by an identifier.
        """
        pattern = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\([^()]*\)')
        prev = None
        while prev != text:
            prev = text
            text = pattern.sub('', text)
        return text

    def check_boolean_precedence(self, cypher_query: str):
        """
        Cypher (like SQL) binds AND tighter than OR. A WHERE clause with
        unparenthesized "X AND Y OR Z" therefore parses as "(X AND Y) OR Z",
        not "X AND (Y OR Z)" -- a silent logic bug, not a syntax error.
        This check flags any WHERE clause containing both AND and OR without
        parentheses grouping them.
        """
        issues = []
        for m in re.finditer(r'\bWHERE\b(.*?)(?:\bWITH\b|\bRETURN\b|$)',
                              cypher_query, re.IGNORECASE | re.DOTALL):
            clause = m.group(1)
            has_and = re.search(r'\bAND\b', clause, re.IGNORECASE)
            has_or  = re.search(r'\bOR\b', clause, re.IGNORECASE)
            # Parens from plain function calls (e.g. toLower(x)) don't count
            # as boolean grouping -- only parens that survive stripping every
            # "word(...)" call do, since a genuine grouping paren is never
            # immediately preceded by an identifier.
            stripped = self._strip_function_call_parens(clause)
            has_parens = "(" in stripped and ")" in stripped
            if has_and and has_or and not has_parens:
                issues.append(
                    "Ambiguous boolean precedence in WHERE clause: contains both AND and OR "
                    "without parentheses. Cypher binds AND tighter than OR, so 'A AND B OR C' "
                    "parses as '(A AND B) OR C', which silently changes the intended meaning. "
                    "Add explicit parentheses to group conditions correctly, e.g. "
                    "'A AND (B OR C)'."
                )
        return issues

    # ---------- Check: comma-separated disconnected patterns within one MATCH ----------
    @staticmethod
    def _split_top_level(text, sep=","):
        """Split `text` on `sep`, but only at bracket/paren/brace depth 0."""
        parts, buf, depth = [], [], 0
        for ch in text:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == sep and depth == 0:
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        parts.append("".join(buf))
        return parts

    def check_match_clause_cartesian(self, cypher_query: str):
        """
        A single MATCH clause can list multiple comma-separated patterns, e.g.
            MATCH (a:A)-[:R1]->(b:B), (c:C)-[:R2]->(d:D)
        If none of the patterns share a variable with any other pattern in the
        SAME clause, Neo4j computes their Cartesian product -- silently
        returning every combination of matches from each independent pattern,
        not a genuine joint traversal. Unlike check_connectivity (which only
        looks at WITH-separated clauses), this catches disconnection WITHIN a
        single MATCH clause, which produces no WITH-carried variable to flag
        and so slips past that check entirely.
        """
        issues = []
        match_clause_matches = list(re.finditer(
            r'\bMATCH\b(.*?)(?=\bMATCH\b|\bWITH\b|\bWHERE\b|\bRETURN\b|$)',
            cypher_query, re.IGNORECASE | re.DOTALL))
        for pos, clause_match in enumerate(match_clause_matches):
            # Only the LAST MATCH clause is checked: an earlier clause's
            # disconnected patterns may still be legitimately reconciled by a
            # later MATCH (e.g. "MATCH (p),(t) MATCH (p)-[:R]->(t)"), which
            # produces correct results despite the intermediate Cartesian
            # step. Only a disconnection with no subsequent MATCH to bridge
            # it is guaranteed to leak into the final RETURN.
            if pos != len(match_clause_matches) - 1:
                continue
            clause_body = clause_match.group(1)
            top_level_parts = self._split_top_level(clause_body)
            if len(top_level_parts) < 2:
                continue
            var_sets = []
            for part in top_level_parts:
                vars_in_part = set(re.findall(r'\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[:){]', part))
                if vars_in_part:
                    var_sets.append(vars_in_part)
            if len(var_sets) < 2:
                continue
            # Union-find over parts that share at least one variable.
            parent = list(range(len(var_sets)))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for i in range(len(var_sets)):
                for j in range(i + 1, len(var_sets)):
                    if var_sets[i] & var_sets[j]:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[ri] = rj
            components = {find(i) for i in range(len(var_sets))}
            if len(components) > 1:
                issues.append(
                    "Cartesian-product risk: this MATCH clause lists multiple comma-separated "
                    "patterns that share NO common variable with each other, e.g. "
                    "'MATCH (a)-[:R1]->(b), (c)-[:R2]->(d)' where {a,b} and {c,d} never overlap. "
                    "Neo4j will return every combination of matches from each independent "
                    "pattern, not a real joint traversal -- this is the same bug the "
                    "WITH-disconnection check catches, just within a single MATCH clause instead "
                    "of across clauses. Connect the patterns via a shared variable, or split into "
                    "separate MATCH clauses joined by a real relationship."
                )
        return issues

    def check_cypher(self, cypher_query: str):
        """Backward-compatible entry point matching the original API."""
        # Arrow-syntax check runs first: if the arrows are malformed, the
        # ontology-path regex below cannot reliably recognize edges at all,
        # so its results would be misleading until this is fixed first.
        issues = self.check_arrow_syntax(cypher_query)
        if issues:
            return False, " | ".join(issues)

        issues = self.check_ontology(cypher_query)
        issues += self.check_property_names(cypher_query)
        issues += self.check_variable_scope(cypher_query)
        issues += self.check_connectivity(cypher_query)
        issues += self.check_match_clause_cartesian(cypher_query)
        issues += self.check_boolean_precedence(cypher_query)
        if issues:
            return False, " | ".join(issues)
        return True, "Valid"


# Backward-compatible alias so existing references (including in the paper
# text) still resolve, while the implementation now covers more than regex
# ontology checks alone.
RegexOntologyChecker = QueryStructureValidator


# ==========================================
# 4. Text-to-Cypher chain
# ==========================================
_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "cypher_prompt.txt"


def _looks_like_refusal(cypher_text: str) -> bool:
    """
    Detects Cypher outputs that are actually a refusal/no-path declaration
    rather than an executable query -- whether or not they follow the exact
    'UNANSWERABLE:' sentinel convention. This matters because a refusal
    phrased as a syntactically valid Cypher literal (e.g.
    "RETURN 'No valid path in schema' AS Error") passes every structural
    check trivially (there is no relationship pattern to validate) and
    would otherwise be executed, producing a nonsense "answer" instead of
    being routed to the unanswerable-handling branch.
    """
    text = cypher_text.strip().upper()
    if text.startswith("UNANSWERABLE"):
        return True
    # A query with no MATCH clause at all that just returns a literal
    # string is almost certainly a disguised refusal, not a real query.
    if "MATCH" not in text and "RETURN" in text:
        refusal_phrases = [
            "NO VALID PATH", "CANNOT BE ANSWERED", "NOT ANSWERABLE",
            "NO PATH IN SCHEMA", "SCHEMA DOES NOT SUPPORT",
            "NO RELATIONSHIP", "CANNOT BE DETERMINED",
        ]
        if any(phrase in text for phrase in refusal_phrases):
            return True
    return False


_RP_NODE = r'\(\s*[A-Za-z0-9_]+\s*(?::\s*[A-Za-z0-9_]+)?[^)]*\)'
_RP_BR   = r'\[[^\]]*?:[A-Za-z0-9_]+[^\]]*\]'
_RP_PAT  = re.compile(rf'({_RP_NODE})\s*(<-|-)\s*({_RP_BR})\s*(->|-)\s*({_RP_NODE})')


def _repair_cypher(cypher, validator):
    """
    Deterministic pre-validation repair of two mechanical LLM mistakes that
    otherwise burn retries (and can hang the run when the model repeats them):
      1. Malformed reversed arrow written on the wrong side of the bracket,
         e.g. '(a)-[:REL]<-(b)'  ->  '(a)<-[:REL]-(b)'.
      2. Relationship direction that is invalid in the ontology but valid when
         reversed -> flip the arrows to the valid direction.
    Only flips when both endpoints are labeled and the reverse pair is a
    declared valid_triplet, so valid queries are never altered.
    Returns (repaired_cypher, notes).
    """
    notes = []
    c = re.sub(r'-(\[[^\]]*\])<-', r'<-\1-', cypher)   # )-[:R]<-(  ->  )<-[:R]-(
    c = re.sub(r'->(\[[^\]]*\])-', r'-\1->', c)         # )->[:R]-(  ->  )-[:R]->(
    if c != cypher:
        notes.append("normalized malformed reversed arrow")
    cypher = c

    var2label = dict(re.findall(r'\(\s*([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)', cypher))
    valid = validator.valid_triplets

    def _flip(m):
        nl, la, br, ra, nr = m.groups()
        lv = re.match(r'\(\s*([A-Za-z0-9_]+)', nl).group(1)
        rv = re.match(r'\(\s*([A-Za-z0-9_]+)', nr).group(1)
        rel = re.search(r':([A-Za-z0-9_]+)', br).group(1)
        ll, rr = var2label.get(lv), var2label.get(rv)
        if not ll or not rr:
            return m.group(0)
        if la == '-' and ra == '->' and (ll, rel, rr) not in valid and (rr, rel, ll) in valid:
            notes.append(f"flipped ({ll})-[:{rel}]->({rr}) to valid direction")
            return f"{nl}<-{br}-{nr}"
        if la == '<-' and ra == '-' and (rr, rel, ll) not in valid and (ll, rel, rr) in valid:
            notes.append(f"flipped ({rr})-[:{rel}]->({ll}) to valid direction")
            return f"{nl}-{br}->{nr}"
        return m.group(0)

    cypher = _RP_PAT.sub(_flip, cypher)
    return cypher, notes


def build_cypher_chain(llm_model="gpt-4.1", base_url=None, api_key=None):
    """llm_model/base_url let the NL->Cypher generator be swapped for a local
    vLLM-served model instead of OpenAI's API -- point base_url at an
    OpenAI-compatible server (e.g. `vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ`,
    default port 8000 -> base_url="http://localhost:8000/v1") and pass that
    server's served-model name as llm_model. api_key is ignored by vLLM's
    server but required by the client library, so it defaults to "EMPTY" for
    any non-default base_url."""
    cypher_system = _PROMPT_FILE.read_text(encoding="utf-8")
    # seed pins the model's backend sampling as much as OpenAI's API allows --
    # temperature=0 alone is NOT a determinism guarantee (confirmed empirically:
    # ~38% of generated queries differed between two otherwise-identical runs
    # without a seed). Not a perfect guarantee even with seed, but meaningfully
    # reduces run-to-run churn per OpenAI's own documentation. vLLM servers
    # generally respect temperature=0 + seed the same way.
    llm_kwargs = dict(model=llm_model, temperature=0, seed=42, timeout=60, max_retries=3)
    if base_url:
        llm_kwargs["base_url"] = base_url
        llm_kwargs["api_key"] = api_key or "EMPTY"
    llm = ChatOpenAI(**llm_kwargs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", cypher_system),
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


def _is_null_only_result(db_output: str) -> bool:
    """True if every row of the (stringified) query result carries no positive
    information -- i.e. every field value is None or False, regardless of the
    field's name. Catches cases like [{'p.personEmail': None}] that a fixed
    'result' key check would miss (the KG matched a node but the requested
    property is unset, which is equivalent to "no data")."""
    try:
        parsed = ast.literal_eval(db_output)
    except (ValueError, SyntaxError):
        return False
    if not isinstance(parsed, list) or not parsed:
        return False
    for row in parsed:
        if not isinstance(row, dict) or not row:
            return False
        if any(v not in (None, False) for v in row.values()):
            return False
    return True


def evaluate_utility(eval_llm, question: str, db_output: str, gold_ans: str) -> EvaluationResult:
    gold_empty = "not available" in str(gold_ans).lower() or str(gold_ans).strip() == ""
    db_l = db_output.lower()
    db_empty   = (
        db_output in ("[]", "")
        or "not available" in db_l
        or "could not be extracted" in db_l
        or "unanswerable" in db_l
        or "error during execution" in db_l
        # a purely boolean-negative / all-null row carries no positive information,
        # regardless of which field(s) it's on
        or _is_null_only_result(db_output)
    )

    # --- deterministic short-circuits: skip the (non-deterministic) LLM judge on
    #     unambiguous cases so identical inputs always score the same way. ---
    if gold_empty and db_empty:
        return EvaluationResult(is_useful=True, reason="Both correctly identified missing information.")
    if gold_empty and not db_empty:
        # gold says the fact is absent from the corpus, yet the KG returned content
        # -> fabrication, always a failure (was a source of judge flip-flopping).
        return EvaluationResult(is_useful=False,
                                reason="Gold is 'not available' but the KG returned content (fabrication).")

    prompt = (
        f"You are evaluating the usability of a Knowledge Graph Question Answering system.\n"
        f"User Question: {question}\n"
        f"Expected Gold Answer: {gold_ans}\n"
        f"Raw Database Output: {db_output}\n"
        f"Does the raw database output contain the correct information to satisfy the user's "
        f"question compared to the Gold Answer?\n\n"
        f"SCORING RULE for list/enumeration answers (e.g. 'who are the collaborators', 'which "
        f"metrics are used'): judge on RECALL of the gold items, not exact-set equality.\n"
        f"  - If the raw output is MISSING one or more of the gold items, mark is_useful=False -- "
        f"that is a genuine gap.\n"
        f"  - If the raw output contains ALL of the gold items but ALSO contains extra items not "
        f"listed in gold, still mark is_useful=True. A knowledge graph legitimately may hold "
        f"additional correct facts beyond what one hand-curated gold answer enumerates, and a user "
        f"reading the output would still find the complete correct answer inside it. Only treat "
        f"extra items as disqualifying if the question demands a single, exclusive answer (e.g. "
        f"'what IS the title', 'who is THE corresponding author') where multiple conflicting "
        f"candidates make the true answer ambiguous."
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
    validator = QueryStructureValidator(ontology_data)

    cypher_chain = build_cypher_chain()
    eval_llm     = ChatOpenAI(model="gpt-5.1", temperature=0, seed=42, timeout=60, max_retries=3)

    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    tag_part = f"_{args.tag}" if args.tag else ""
    os.makedirs(args.outdir, exist_ok=True)
    output_csv = os.path.join(args.outdir, f"KG_UTILITY_RESULTS_{prefix}_{input_stem}{tag_part}.csv")
    output_log = os.path.join(args.outdir, f"KG_UTILITY_LOG_{prefix}_{input_stem}{tag_part}.txt")

    df = pd.read_csv(args.input)

    db_outputs, generated_cyphers, eval_results, eval_reasons, failure_tags = [], [], [], [], []

    with open(output_log, "w", encoding="utf-8") as log_f:

        def log(text):
            print(text)
            log_f.write(text + "\n")
            log_f.flush()

        log(f"Evaluating {prefix} KG on {len(df)} questions using QueryStructureValidator...")

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

                    # Deterministic repair of mechanical arrow/direction mistakes
                    # BEFORE validation, so they don't burn retries or hang the run.
                    gen_cypher, repair_notes = _repair_cypher(gen_cypher, validator)
                    if repair_notes:
                        log(f"  [Attempt {attempt}] Auto-repaired: {'; '.join(repair_notes)}")

                    # Check 1: structural/ontology/connectivity/precedence
                    struct_valid, struct_reason = validator.check_cypher(gen_cypher)
                    if not struct_valid:
                        feedback_history.append(struct_reason)
                        failure_tag = "ontology_or_structure"
                        log(f"  [Attempt {attempt}] Structural check failed: {struct_reason}")
                        continue

                    # Check 2: raw syntax pre-check via EXPLAIN (no execution)
                    syntax_valid, syntax_reason = explain_check(graph, gen_cypher)
                    if not syntax_valid:
                        feedback_history.append(f"Cypher syntax error (via EXPLAIN): {syntax_reason}")
                        failure_tag = "syntax_error"
                        log(f"  [Attempt {attempt}] EXPLAIN syntax check failed: {syntax_reason}")
                        continue

                    # Check 3: trial execution. Some queries pass EXPLAIN but throw at
                    # runtime on real data (e.g. date() on a non-ISO value, scoping).
                    # Feed the actual Neo4j error back so the LLM can self-correct.
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
                # db_output_str was captured during the successful trial execution (Check 3) above.
                # Relaxation retry: a VALID query that returned ZERO rows, OR a boolean existence
                # check (RETURN count(x) > 0 AS result) that came back False, is often
                # OVER-CONSTRAINED -- e.g. ANDing several literal phrase-fragments from the question
                # onto one property when the graph's actual text differs slightly -- even though the
                # fact exists in the KG. Try ONE less-restrictive query in either case. Gold-agnostic:
                # the model may return UNANSWERABLE if no faithful relaxation exists, so truly-absent
                # facts are not forced into answers.
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
                        if not _looks_like_refusal(rc) and validator.check_cypher(rc)[0] \
                                and explain_check(graph, rc)[0]:
                            rraw = graph.query(rc)
                            rraw_str = str(rraw)
                            if rraw_str not in ("[]", "") and not _is_null_only_result(rraw_str):
                                db_output_str = rraw_str
                                gen_cypher = rc
                                failure_tag = "none"
                                log("  [Relaxation] over-constrained query relaxed -> rows found")
                    except Exception as e:
                        log(f"  [Relaxation] skipped ({e})")
            # Truncate only a JUDGE-PROMPT copy, never the stored db_output_str
            # itself -- deterministic_eval.py later re-parses the same
            # DB_OUTPUT CSV column with ast.literal_eval, and a value cut off
            # mid-token (as the old in-place truncation did) is permanently
            # unparseable, silently scoring genuinely-correct rows as "empty."
            db_output_for_judge = db_output_str
            if len(db_output_str) > 3000:
                log("Warning: Massive DB result detected. Truncating for evaluator prompt only "
                    "(full result still stored for grading)...")
                db_output_for_judge = db_output_str[:3000] + "... [TRUNCATED FOR JUDGE PROMPT]"

            try:
                result = evaluate_utility(eval_llm, question, db_output_for_judge, gold_ans)
            except Exception as e:
                log(f"Evaluator API Error: {e}")
                result = EvaluationResult(is_useful=False, reason="API Error: evaluation failed.")

            if result.is_useful and failure_tag not in ("none", "unanswerable_by_schema"):
                failure_tag = "none"  # recovered via self-repair loop

            db_outputs.append(db_output_str)
            generated_cyphers.append(gen_cypher)
            eval_results.append(result.is_useful)
            eval_reasons.append(result.reason)
            failure_tags.append(failure_tag)

            log(f"Cypher: {gen_cypher}")
            log(f"DB Output: {db_output_str[:100]}...")
            log(f"Failure tag: {failure_tag}")
            log(f"Eval: {'Pass' if result.is_useful else 'Fail'} ({result.reason})")

            time.sleep(1)

        df["GENERATED_CYPHER"] = generated_cyphers
        df["DB_OUTPUT"]        = db_outputs
        df["IS_USEFUL"]        = eval_results
        df["EVAL_REASON"]      = eval_reasons
        df["FAILURE_TAG"]      = failure_tags
        df.to_csv(output_csv, index=False)

        accuracy = sum(eval_results) / len(eval_results) * 100
        log(f"\n====================================")
        log(f"{prefix} Pipeline Evaluation Complete! Accuracy: {accuracy:.2f}%")
        log("Failure taxonomy breakdown:")
        for tag, count in pd.Series(failure_tags).value_counts().items():
            log(f"  {tag}: {count}")
        log(f"Data saved to {output_csv}")
        log(f"Logs saved to {output_log}")
        log(f"====================================")


if __name__ == "__main__":
    main()
