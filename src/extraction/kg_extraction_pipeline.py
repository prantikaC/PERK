# -*- coding: utf-8 -*-
"""
Unified KG extraction pipeline for PERK.

Open-source models (Gemma, LLaMA, Qwen 7B/32B) run locally with in-process vLLM;
gpt-oss-20b runs via a local vLLM OpenAI-compatible server; GPT-4.1 / GPT-5.1 run
via the OpenAI API. (GPT-5.1 is the only model that requires an API call.)
"""

import os
import sys
import csv
import json
import glob
import time
import logging
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm.auto import tqdm
import pandas as pd


# Pin GPU(s) BEFORE importing torch/vLLM — both read CUDA_VISIBLE_DEVICES once, at
# import time. We parse --gpu from argv here and force PCI-bus ordering so --gpu N is
# physical GPU N (the number nvidia-smi shows). Accepts a comma list for tensor
# parallelism, e.g. --gpu 0,1.
def _pin_gpu_from_argv():
    gpu = os.environ.get("PERK_GPU")
    for i, a in enumerate(sys.argv):
        if a == "--gpu" and i + 1 < len(sys.argv):
            gpu = sys.argv[i + 1]
        elif a.startswith("--gpu="):
            gpu = a.split("=", 1)[1]
    if gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return gpu

_PINNED_GPU = _pin_gpu_from_argv()

try:
    from dotenv import load_dotenv
    load_dotenv()
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ==============================================================================
# CONSTANTS
# ==============================================================================

# src/extraction/ -> src/ -> prompts/
DEFAULT_PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "extraction_prompt.txt"

# Open-source models run locally with in-process vLLM.
VLLM_MODELS = {"gemma", "llama", "qwen", "qwen32b"}
# OpenAI / OpenAI-compatible endpoint. gptoss = gpt-oss-20b served by a local vLLM
# OpenAI-compatible server (vLLM handles the harmony formatting); openai = GPT-4.1/5.1.
API_MODELS = {"openai", "gptoss"}

# Default model name per alias when --model_path is omitted.
DEFAULT_MODEL_PATHS = {"gptoss": "openai/gpt-oss-20b"}
# Default OpenAI-compatible endpoint per alias when --base_url is omitted.
DEFAULT_BASE_URLS = {"gptoss": "http://localhost:8000/v1"}   # local vLLM server

HEADER_EVIDENCE = "Header Metadata"

PREFIX_MAP = {
    "Person": "pn", "Email": "e", "MailThread": "t",
    "Paper": "pa", "PaperBib": "pb", "Conference": "c",
    "Journal": "j", "Dataset": "d", "Method": "me",
    "Task": "tk", "Metric": "mt", "Meeting": "mg",
    "SubmissionID": "s", "PaperStatus": "ps",
}

# Entity identity category, per type (see PERK_v2/README.md / data/ledger.json
# entity_type_policy for the full rationale). This governs how
# get_canonical_key/create_unique_entity_id decide whether a newly-extracted
# mention refers to an EXISTING entity (reuse its ID) or is a genuinely NEW
# one (mint a fresh ID):
#   fixed           - one immutable record, minted once, referenced forever.
#                     Dedup by a stable identifying field (name/email/title/...).
#   series_instance - a recurring template (e.g. "JCDL") spawns independent,
#                      parallel instances distinguished by edition/date.
#                      Dedup by identifying field + edition-distinguishing
#                      field where available, so different editions never
#                      collide onto one node.
#   snapshot_chain   - an append-only sequence of immutable snapshots
#                       (PaperStatus). Every mention is a NEW node by
#                       definition -- there is no lookup-by-key at all.
# Root cause this fixes: get_canonical_key previously keyed PaperStatus on
# `statusType` alone (e.g. "Submitted"), so every "Submitted" status across
# every unrelated submission in the whole corpus resolved to the SAME
# canonical key and got fused into one entangled node at extraction time,
# before entity resolution ever ran.
ENTITY_TYPE_POLICY = {
    "Person": "fixed", "Dataset": "fixed", "Method": "fixed", "Metric": "fixed",
    "Paper": "fixed", "SubmissionID": "fixed", "PaperBib": "fixed",
    "Task": "fixed", "Journal": "fixed",
    "Conference": "series_instance", "Meeting": "series_instance",
    "PaperStatus": "snapshot_chain",
    "Email": "output_layer", "MailThread": "output_layer",
}

# For series_instance types, the field (if present on this mention) that
# distinguishes one edition/occurrence from another with the same name.
SERIES_EDITION_FIELD = {
    "Conference": "confDate",
    "Meeting": "meetDate",
}

# Vague, non-distinguishing names that sometimes get extracted for a "fixed"
# entity when an email refers to it informally instead of using its real,
# specific title/name -- e.g. an email saying "the revised manuscript is
# ready" produces paperTitle="revised manuscript" instead of the paper's
# actual title. Confirmed live-graph audit evidence for why treating these as
# a stable dedup key is actively harmful, not just imprecise: a single Paper
# node titled "JOCCH minor revision" ended up `identifies`'d by SubmissionIDs
# from two different years -- i.e. two genuinely different real papers, each
# vaguely described this way in some email, were silently merged into one.
# Likewise a Dataset node literally named "datasets" existed as its own
# entity. For names on this list, never dedupe by string match -- mint a
# fresh ID every time (same mechanism as snapshot_chain types), since an
# unmerged duplicate is far easier to catch and fix than a false merge.
# Exact-match only: bare generic nouns that are never themselves a real,
# specific Paper/Dataset/etc. name.
GENERIC_NAME_DENYLIST_EXACT = {
    "dataset", "datasets", "document", "documents", "method", "methods",
    "metric", "metrics", "task", "tasks", "paper", "papers", "manuscript",
}

# Substring match: vague informal REFERENCE phrases that stay generic no
# matter what venue/context name gets prefixed onto them -- e.g. an email's
# ad hoc "JOCCH minor revision" is exactly as unreliable an identity signal
# as bare "minor revision" would be; the venue name doesn't make it specific
# to one paper (confirmed: this exact phrase collided two different papers).
# "submission" bare (not just "our submission"/"the submission") is required
# to catch "<venue> submission" phrasing (e.g. "JOCCH submission") -- confirmed
# via live-graph audit that this exact phrase, extracted verbatim across four
# different real submissions spanning 2022-2025, collapsed all four onto one
# Paper node (pa23) despite each having a distinct, non-colliding SubmissionID.
GENERIC_NAME_DENYLIST_SUBSTRING = {
    "the paper", "the manuscript", "the dataset", "the task", "the method",
    "submission", "revised manuscript",
    "camera-ready manuscript", "minor revision",
}


def _is_generic_placeholder_name(name: str) -> bool:
    """True if `name` is a vague placeholder rather than a real, specific
    identifying name (see GENERIC_NAME_DENYLIST_EXACT/_SUBSTRING)."""
    normalized = name.lower().strip()
    if normalized in GENERIC_NAME_DENYLIST_EXACT:
        return True
    return any(phrase in normalized for phrase in GENERIC_NAME_DENYLIST_SUBSTRING)


# Targeted Rule 2 (Appendix B.1): a person merely mentioned near a paper --
# most often in an email salutation or sign-off -- must not be misattributed
# as an author. Applied here, on the merged raw hasAuthor edges, before
# entity resolution: 98.8% of raw hasAuthor edges pointed to just the small
# set of participants who appear most often across the whole corpus,
# regardless of whether a given email actually said anything about
# authorship, which is exactly why every extracted triple carries an
# evidence sentence in the first place -- so this check can run on it.
HASAUTHOR_POSITIVE_PATTERNS = [
    re.compile(r"\bco-?authors?\b", re.I),
    re.compile(r"\bour\b[^.]{0,40}\b(manuscript|paper|submission|proposal)\b", re.I),
    re.compile(r"\bcorresponding author\b", re.I),
    re.compile(r"\bauthored by\b", re.I),
    re.compile(r"\bauthor of\b", re.I),
    re.compile(r"\bsubmitted by\b", re.I),
    re.compile(r"\bon behalf of my co-authors\b", re.I),
    re.compile(r"\bjoint contributions?\b", re.I),
    re.compile(r"\bauthorship\b", re.I),
    re.compile(r"\bauthors?:\s", re.I),
    re.compile(r"-(?:Lead|Equal|Supporting)\b"),
]


def _has_hasauthor_evidence(context: str) -> bool:
    """True if `context` (the evidence text captured for a hasAuthor triple)
    contains real authorship-supporting language, not just a person mentioned
    nearby (e.g. an email salutation or sign-off). CONTAINMENT check: an edge
    whose context is a salutation FOLLOWED BY "our manuscript..." in the same
    captured snippet is correctly kept."""
    text = str(context or "")
    return any(p.search(text) for p in HASAUTHOR_POSITIVE_PATTERNS)

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("KG_Extraction")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter('%(message)s')
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model(args):
    """
    Load the inference backend. Returns (engine, sampling, client):
      - API models        -> (None, None, OpenAI client)
      - local vLLM models  -> (vllm.LLM, vllm.SamplingParams, None)
    """
    model_type = args.model.lower()

    if model_type in API_MODELS:
        if not OPENAI_AVAILABLE:
            raise ImportError("pip install openai python-dotenv")
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = args.base_url or DEFAULT_BASE_URLS.get(model_type)
        if base_url:
            # Local OpenAI-compatible server (e.g. vLLM serving gpt-oss-20b).
            # The server handles harmony formatting; key may be a dummy.
            print(f"Using OpenAI-compatible endpoint: {base_url}")
            return None, None, OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        return None, None, OpenAI(api_key=api_key)

    # Local open-source model via in-process vLLM
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("pip install vllm")

    if _PINNED_GPU is not None:
        print(f"Pinned to GPU(s): {_PINNED_GPU} (CUDA_DEVICE_ORDER=PCI_BUS_ID)")
    print(f"Loading {args.model_path} with vLLM "
          f"(tensor_parallel_size={args.tensor_parallel_size})...")

    engine = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    # Greedy decoding for reproducible extraction.
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    print("Model loaded.")
    return engine, sampling, None

# ==============================================================================
# INFERENCE
# ==============================================================================

def run_inference(messages: List[Dict], engine, sampling, client, args) -> str:
    model_type = args.model.lower()

    if model_type in API_MODELS:
        kwargs = dict(
            model=args.model_path,
            messages=messages,
            response_format={"type": "json_object"},
            # seed works across all chat-completions models (unlike temperature,
            # which the gpt-5 family rejects at anything but its default of 1) --
            # doesn't guarantee determinism, but meaningfully reduces run-to-run
            # variance for the same reason it was added to the Cypher-generation
            # LLM calls in kg_eval_v4.py.
            seed=42,
        )
        # The gpt-5 family only accepts the default temperature (1); sending 0
        # errors. Other models (e.g. gpt-4.1) use 0 for deterministic output.
        if not re.search(r"gpt-5", str(args.model_path), re.IGNORECASE):
            kwargs["temperature"] = 0
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    # Local vLLM: chat() applies the model's chat template automatically.
    outputs = engine.chat(messages, sampling, use_tqdm=False)
    return outputs[0].outputs[0].text

# ==============================================================================
# REGEX HEADER PARSING
# ==============================================================================

# Matches display names like "ACM", "ACM 2019", "ACL2024", "JOCCH" -- a
# short all-caps acronym optionally followed by a 4-digit year, with no
# lowercase anywhere. Real human names never fit this shape (they always
# have lowercase letters), so this catches organizational senders that DO
# carry a display name (e.g. "ACM 2019 <acm2019@acm.com>"), which a bare-
# email check alone would miss (that only fires when there is NO name).
ORG_DISPLAY_NAME_PATTERN = re.compile(r'^[A-Z]{2,8}\s*\d{0,4}$')


def parse_email_participants(text: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Parse a From/To/Cc header field. Returns (persons, org_email_hints).

    A header address is treated as ORGANIZATIONAL, not a Person, when either:
      - there is NO accompanying display name at all (a bare
        "acl2024-submissions@aclconference.org" in a To: line, as opposed to
        "Name <email>"), or
      - the display name matches ORG_DISPLAY_NAME_PATTERN (e.g. "ACM 2019").
    Conference/journal submission systems and editorial-office mailboxes
    show up in headers exactly one of these two ways. Root cause this
    avoids: such addresses were previously always minted as Person nodes
    with only a personEmail property (or a nonsensical "personName" like
    "ACM 2019") -- silently polluting the Person type with non-human
    entities.

    These organizational addresses are NOT extracted as any entity here --
    there is no Organization node type. They are returned as plain email
    strings (org_email_hints) so the caller can surface them to the
    body-extraction LLM as candidate journalMail/confMail values for
    whatever Journal/Conference it independently extracts from the body.
    """
    persons, org_email_hints = [], []
    for part in re.split(r'[,;]', text.strip()):
        part = part.strip()
        if not part:
            continue
        name = email = affiliation = None
        match = re.search(r'(.+?)\s*[<\[]([^>\]]+)[>\]]', part)
        if match:
            name, email = match.group(1).strip(), match.group(2).strip()
        else:
            match = re.search(r'(.+?)\s*\(([^)]+)\)', part)
            if match:
                name = match.group(1).strip()
                potential = match.group(2).strip()
                if '@' in potential:
                    email = potential
                else:
                    affiliation = potential
            else:
                if '@' in part:
                    email = part
                else:
                    name = part

        if email and not name:
            org_email_hints.append(email)
            continue

        if email and name and ORG_DISPLAY_NAME_PATTERN.match(name.strip()):
            org_email_hints.append(email)
            continue

        person = {}
        if name: person['personName'] = name.replace('"', '').strip()
        if email: person['personEmail'] = email
        if affiliation: person['affiliation'] = affiliation
        if person:
            persons.append(person)
    return persons, org_email_hints


def parse_date(date_str: str) -> str:
    date_str = date_str.strip()
    match = re.match(r'(\d{1,2})-(\d{1,2})-(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return date_str


def extract_header_info(email_text: str) -> Dict:
    """
    All keyword patterns below are anchored with ^ (used with re.MULTILINE)
    so the keyword must be the FIRST thing on its own line -- see
    header_signature_parser.py's identical fix for why: an unanchored
    r'To:\\s*...' matches "To:" wherever it first occurs, including inside a
    "reply-To:" header line (real in Gmail-exported mail, never present in
    synthetic PATRA.txt), which cascades into garbled to/cc/date/subject
    values for that whole email.
    """
    header_info = {}
    for field, pattern in [
        ('thread_id', r'^Thread ID:\s*(.+)'),
        ('mail_id',   r'^Mail ID:\s*(.+)'),
    ]:
        match = re.search(pattern, email_text, re.IGNORECASE | re.MULTILINE)
        if match:
            header_info[field] = match.group(1).strip()

    match = re.search(r'^Date:\s*(.+)', email_text, re.IGNORECASE | re.MULTILINE)
    if match:
        header_info['date'] = parse_date(match.group(1).strip())

    # Generic "looks like a header label" stop-boundary, not a hardcoded
    # From->To->Cc->Subject order -- see header_signature_parser.py's
    # identical fix for why (e.g. "reply-To:" between From and To, "Date:"
    # between CC and Subject; a fixed field-name enumeration would miss
    # whatever a real email client adds that PATRA.txt never anticipated).
    _header_labels = r'(?:[A-Za-z][A-Za-z \-]*:)'
    for field, pattern in [
        ('from',    r'^From:\s*(.+?)(?=\n' + _header_labels + r'|\Z)'),
        ('to',      r'^To:\s*(.+?)(?=\n' + _header_labels + r'|\Z)'),
        ('cc',      r'^Cc:\s*(.+?)(?=\n' + _header_labels + r'|\Z)'),
    ]:
        match = re.search(pattern, email_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if match:
            header_info[field] = match.group(1).strip()

    match = re.search(r'^Subject:\s*(.+)', email_text, re.IGNORECASE | re.MULTILINE)
    if match:
        header_info['subject'] = match.group(1).strip()

    return header_info


def extract_body(email_text: str) -> str:
    """
    Scan for the last header line to find where the body starts.
    Falls back to the first double-newline if no headers are matched.
    """
    header_pattern = re.compile(
        r'^(?:From|To|Cc|Bcc|Date|Subject|Thread-?ID|Mail-?ID|Message-?ID):',
        re.IGNORECASE
    )
    lines = email_text.splitlines()
    body_start_index = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if header_pattern.match(stripped):
            body_start_index = i + 1
        elif line.startswith((' ', '\t')) and i > 0 and body_start_index == i:
            body_start_index = i + 1

    body = "\n".join(lines[body_start_index:]).strip()
    if not body:
        match = re.search(r'\n\s*\n(.+)', email_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return body

# ==============================================================================
# ENTITY MANAGEMENT (module-level state, reset per run via main())
# ==============================================================================

entity_registry: Dict = {}
id_counters: Dict = defaultdict(int)
entities_all: List = []
relations_all: List = []


def get_canonical_key(entity_type: str, properties: dict) -> str:
    if entity_type == "Person":
        email = properties.get("personEmail", "").lower().strip()
        return f"person_email_{email}" if email else f"person_name_{properties.get('personName', '').lower().strip()}"

    # snapshot_chain: never look up by key -- every mention is a new node.
    # (See ENTITY_TYPE_POLICY above; this is what stops e.g. every unrelated
    # submission's "Submitted" status from colliding onto one PaperStatus node.)
    if ENTITY_TYPE_POLICY.get(entity_type) == "snapshot_chain":
        return f"{entity_type.lower()}_{id(properties)}_{time.time_ns()}"

    field_map = {
        "Paper": "paperTitle", "PaperBib": "doi", "Dataset": "datasetName",
        "Method": "methodName", "Task": "taskName", "Metric": "metricName",
        "Email": "mailNum", "MailThread": "threadID", "SubmissionID": "identifier",
        "Conference": "confTitle", "Journal": "journalTitle",
        "PaperStatus": "statusType", "Meeting": "meetAgenda",
    }
    if entity_type in field_map:
        name_value = str(properties.get(field_map[entity_type], '')).strip()
        if _is_generic_placeholder_name(name_value):
            # Never dedupe on a vague placeholder name -- see
            # GENERIC_NAME_DENYLIST for the confirmed false-merge cases this
            # prevents. Mint a fresh ID every time, same as snapshot_chain.
            return f"{entity_type.lower()}_{id(properties)}_{time.time_ns()}"
        base_key = f"{entity_type.lower()}_{name_value.lower()}"
        # series_instance: fold in an edition-distinguishing field (e.g.
        # confDate) when this mention states one, so "JCDL" mentioned across
        # two different years doesn't collide onto a single Conference node.
        edition_field = SERIES_EDITION_FIELD.get(entity_type)
        if edition_field and properties.get(edition_field):
            base_key += f"_{str(properties[edition_field]).lower().strip()}"
        return base_key
    return f"{entity_type.lower()}_{id(properties)}"


def create_unique_entity_id(entity_type: str, properties: dict) -> str:
    """Return existing global ID for a known entity, or register and return a new one."""
    canonical_key = get_canonical_key(entity_type, properties)
    for stable_id, data in entity_registry.items():
        if data.get("key") == canonical_key:
            data["properties"].update(properties)
            return stable_id
    prefix = PREFIX_MAP.get(entity_type, "x")
    id_counters[entity_type] += 1
    stable_id = f"{prefix}{id_counters[entity_type]}"
    entity_registry[stable_id] = {
        "type": entity_type,
        "properties": properties.copy(),
        "key": canonical_key,
    }
    return stable_id


def extract_header_entities(header_info: dict, email_num: int) -> Tuple[List, List, List]:
    entities, relations, org_email_hints = [], [], []

    if 'thread_id' in header_info:
        thread_props = {"threadID": header_info['thread_id']}
        if 'subject' in header_info:
            thread_props["subject"] = header_info['subject']
        thread_id = create_unique_entity_id("MailThread", thread_props)
        entities.append([thread_id, "MailThread", json.dumps(thread_props)])

    if 'mail_id' in header_info:
        email_props = {"mailNum": header_info['mail_id']}
        if 'date' in header_info: email_props["mailDate"] = header_info['date']
        email_id = create_unique_entity_id("Email", email_props)
        entities.append([email_id, "Email", json.dumps(email_props)])
        if 'thread_id' in header_info:
            relations.append([email_id, thread_id, "partOf", HEADER_EVIDENCE, "header"])

    for field in ['from', 'to', 'cc']:
        if field in header_info:
            persons, org_emails = parse_email_participants(header_info[field])
            org_email_hints.extend(org_emails)
            for props in persons:
                person_id = create_unique_entity_id("Person", props)
                entities.append([person_id, "Person", json.dumps(props)])
                if 'mail_id' in header_info:
                    rel = "sentBy" if field == 'from' else "receivedBy"
                    relations.append([email_id, person_id, rel, HEADER_EVIDENCE, "header"])

    return entities, relations, org_email_hints


def build_header_persons_context(header_entities: List, org_email_hints: List[str] = None) -> str:
    """
    Build the prompt context string listing Persons found in headers, so the
    body-extraction LLM can reuse their IDs. Also lists any organizational
    sender/recipient addresses detected in headers (editorial offices,
    submission systems -- e.g. "jocch-office@acm.org") as plain email
    strings, NOT as an entity -- there is no Organization node type. These
    are hints only: if a Journal/Conference extracted from the body
    corresponds to one of them, set that entity's own journalMail/confMail
    property to it.
    """
    persons = [e for e in header_entities if e[1] == 'Person']
    org_email_hints = org_email_hints or []
    if not persons and not org_email_hints:
        return "No persons in headers."
    context = ""
    if persons:
        context += "Persons already extracted from email headers (reuse these IDs):\n"
        for eid, _, _ in persons:
            if eid in entity_registry:
                context += f"- {eid}: {json.dumps(entity_registry[eid]['properties'])}\n"
    if org_email_hints:
        context += ("Organizational sender/recipient addresses in this email's headers "
                    "(NOT persons -- do not create any entity for these directly; if a "
                    "Journal/Conference you extract from the body corresponds to one, "
                    "set that entity's journalMail/confMail property to it instead):\n")
        for addr in org_email_hints:
            context += f"- {addr}\n"
    return context

# ==============================================================================
# JSON PARSING — 2-pass robust parser (handles truncated/malformed LLM output)
# ==============================================================================

def manual_json_repair(json_str: str) -> str:
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

    def replace_inner_quotes(match):
        content = re.sub(r'(?<!\\)"', "'", match.group(1))
        return f': "{content}"'

    return re.sub(r':\s*"(.*?)"(?=\s*[,}\]])', replace_inner_quotes, json_str, flags=re.DOTALL)


def parse_llm_output(response_text: str) -> Tuple[List, List]:
    if not response_text or not response_text.strip():
        return [], []

    text = re.sub(r'```(?:json)?', '', response_text).strip()
    text = re.sub(r'```', '', text).strip()

    entities: List = []
    relations: List = []

    def extract_from_dict(d: dict) -> None:
        if "entities" in d and isinstance(d["entities"], list):
            for e in d["entities"]:
                if not isinstance(e, dict): continue
                eid = str(e.get("id", "")).strip()
                etype = str(e.get("type", "")).strip()
                if eid and etype and eid.lower() != "none":
                    entities.append([eid, etype, json.dumps(e.get("properties", {}))])

        if "relations" in d and isinstance(d["relations"], list):
            for r in d["relations"]:
                if not isinstance(r, dict): continue
                s   = str(r.get("start_id") or r.get("start")  or r.get("source") or "").strip()
                e   = str(r.get("end_id")   or r.get("end")    or r.get("target") or "").strip()
                rel = str(r.get("relation") or r.get("type")   or "").strip()
                ctx = str(r.get("context")  or r.get("evidence") or "").strip()
                if s and e and rel and s != "None" and e != "None":
                    relations.append([s, e, rel, ctx])

    # Pass 1: standard JSON (with truncation repair attempt)
    data = None
    for candidate in [text, text + '}' * max(0, text.count('{') - text.count('}'))]:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if data is not None:
        if isinstance(data, dict):
            extract_from_dict(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict): extract_from_dict(item)
        return entities, relations

    # Pass 2: object scraper fallback (for badly malformed output)
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos] not in ['{', '[']:
            pos += 1
        if pos >= len(text):
            break
        try:
            obj, end_pos = decoder.raw_decode(text, pos)

            def process_obj(item: dict) -> None:
                if "id" in item and "type" in item:
                    eid = str(item.get("id", "")).strip()
                    etype = str(item.get("type", "")).strip()
                    if eid and etype and eid.lower() != "none":
                        entities.append([eid, etype, json.dumps(item.get("properties", {}))])
                elif "start_id" in item or "relation" in item:
                    s   = str(item.get("start_id") or item.get("source") or "").strip()
                    e   = str(item.get("end_id")   or item.get("target") or "").strip()
                    rel = str(item.get("relation") or item.get("type")   or "").strip()
                    ctx = str(item.get("context") or "").strip()
                    if s and e and rel:
                        relations.append([s, e, rel, ctx])

            if isinstance(obj, dict):
                if "entities"  in obj: [process_obj(i) for i in obj["entities"]  if isinstance(i, dict)]
                if "relations" in obj: [process_obj(i) for i in obj["relations"] if isinstance(i, dict)]
                process_obj(obj)
            elif isinstance(obj, list):
                [process_obj(i) for i in obj if isinstance(i, dict)]
            pos = end_pos
        except Exception:
            pos += 1

    return entities, relations

# ==============================================================================
# CSV UTIL
# ==============================================================================

def write_csv(filepath: str, headers: List[str], rows: List[List]) -> None:
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        writer.writerows(rows)

# ==============================================================================
# EMAIL PROCESSING
# ==============================================================================

def process_email(email_num: int, email_text: str, engine, sampling, client,
                  system_prompt: str, args, logger: logging.Logger,
                  header_context_override: str = None,
                  current_email_id_override: str = None,
                  extra_valid_ids: set = None,
                  org_email_hints_override: List[str] = None,
                  context_id_map: Dict[str, str] = None) -> Tuple[int, int]:
    """
    header_context_override / current_email_id_override / extra_valid_ids /
    context_id_map let an external orchestrator (run_full_extraction_pipeline.py)
    supply richer header/signature context -- built by header_signature_parser.py,
    which models Team/Organization/EmailID and signature-derived role/affiliation
    that this module's own regex header parser below does not -- instead of
    the crude header parsing this function computes internally by default.
    When supplied, this function's own header parsing is skipped entirely
    and no header entities/relations are written to this email's per-email
    CSVs (the orchestrator's header pass owns and writes those separately);
    `extra_valid_ids` extends the relation-validity check so an LLM relation
    that reuses one of the orchestrator's header-side ids (as the context
    instructs it to) is still accepted.

    `context_id_map` (e.g. {"HDR_pn6": "pn6"}) is the mechanism that makes
    that reuse actually SAFE: the LLM's own local temp-ids for genuinely NEW
    entities it finds in the body are assigned starting fresh at 1 each
    call, with no memory of what header ids already exist -- on a small
    corpus, that range overlaps the header's own (small) id numbers, so the
    LLM can and does accidentally reuse a bare header id (e.g. "pn7") as
    its own temp-id for something else entirely, silently corrupting any
    relation in the same response that meant to reference the header
    entity. Namespacing header ids with a "HDR_" prefix in the prompt (see
    header_signature_parser.build_llm_context) makes that string one the
    LLM's own ad hoc numbering can never accidentally produce -- so a
    "HDR_xxx" token in this email's LLM output is resolved via this map
    STRAIGHT to the real id "xxx", bypassing temp_to_global entirely,
    before any of the usual local-temp-id resolution below runs.

    Standalone CLI use (no orchestrator) is unaffected -- all parameters
    default to None, which reproduces the exact prior behavior.
    """
    body_text = extract_body(email_text)
    if header_context_override is not None:
        header_entities, header_relations = [], []
        header_context = header_context_override
        org_email_hints = org_email_hints_override or []
    else:
        header_info = extract_header_info(email_text)
        header_entities, header_relations, org_email_hints = extract_header_entities(header_info, email_num)
        header_context = build_header_persons_context(header_entities, org_email_hints)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{header_context}\n\nBody:\n{body_text}\n\nExtract JSON."}
    ]

    try:
        response_text = run_inference(messages, engine, sampling, client, args)
    except Exception as e:
        logger.error(f"Inference error on Email {email_num}: {e}")
        response_text = "{}"

    if args.save_debug:
        with open(os.path.join(args.output_dir, f"debug_{args.model}_email{email_num}.txt"), 'w', encoding='utf-8') as f:
            f.write(response_text)

    entities_raw, relations_raw = parse_llm_output(response_text)

    body_entities: List = []
    body_relations: List = []
    context_id_map = context_id_map or {}
    temp_to_global: Dict = {}

    for temp_id, etype, props_str in entities_raw:
        if temp_id in context_id_map:
            # LLM redundantly re-declared an already-existing header entity
            # as if it were new -- alias its temp id straight to the real
            # one instead of minting a duplicate.
            temp_to_global[temp_id] = context_id_map[temp_id]
            continue
        try:
            props = json.loads(props_str)
            global_id = create_unique_entity_id(etype, props)
            temp_to_global[temp_id] = global_id
            row = [global_id, etype, props_str]
            if global_id not in {e[0] for e in entities_all}: entities_all.append(row)
            if global_id not in {e[0] for e in body_entities}: body_entities.append(row)
        except Exception:
            continue

    # Deterministic org-mail attachment: if this email had exactly one
    # organizational address hint and exactly one Journal/Conference was
    # extracted from its body, the match is unambiguous by construction --
    # assign it directly rather than relying on the LLM to notice and set it.
    # Left alone (for the LLM to judge, per rule 3b) whenever there's more
    # than one hint or more than one venue in the same email.
    if len(org_email_hints) == 1:
        venue_rows = [row for row in body_entities if row[1] in ("Journal", "Conference")]
        if len(venue_rows) == 1:
            v_id, v_type, v_props_str = venue_rows[0]
            v_props = json.loads(v_props_str)
            mail_field = "journalMail" if v_type == "Journal" else "confMail"
            if not v_props.get(mail_field):
                v_props[mail_field] = org_email_hints[0]
                new_props_str = json.dumps(v_props)
                venue_rows[0][2] = new_props_str
                if v_id in entity_registry:
                    entity_registry[v_id]["properties"][mail_field] = org_email_hints[0]
                for row in entities_all:
                    if row[0] == v_id:
                        row[2] = new_props_str
                        break

    current_email_id = current_email_id_override or next(
        (e[0] for e in header_entities if e[1] == "Email"), "unknown")

    extra_valid_ids = extra_valid_ids or set()
    for s_id, o_id, rel, ctx in relations_raw:
        # context_id_map takes priority: a "HDR_xxx" token is ALWAYS a
        # direct reference to the real header id "xxx", never something
        # temp_to_global should resolve (see this function's docstring).
        s_glob = context_id_map.get(s_id) or temp_to_global.get(s_id, s_id)
        o_glob = context_id_map.get(o_id) or temp_to_global.get(o_id, o_id)
        valid_start = (s_glob in entity_registry or s_glob in {e[0] for e in body_entities}
                       or s_glob in extra_valid_ids)
        valid_end   = (o_glob in entity_registry or o_glob in {e[0] for e in body_entities}
                       or o_glob in extra_valid_ids)
        if valid_start and valid_end:
            row = [s_glob, o_glob, rel, ctx, current_email_id]
            body_relations.append(row)
            if tuple(row) not in {tuple(r) for r in relations_all}:
                relations_all.append(row)

    all_ents = header_entities + body_entities
    all_rels = header_relations + body_relations

    n_re, n_rl = len(header_entities), len(header_relations)
    n_le, n_ll = len(body_entities), len(body_relations)
    log_msg = (f"Email {email_num}: Regex(Ent={n_re}, Rel={n_rl}) | "
               f"LLM(Ent={n_le}, Rel={n_ll}) | Total(Ent={len(all_ents)}, Rel={len(all_rels)})")

    if n_le == 0 and n_ll == 0:
        logger.warning(f"{log_msg} - EMPTY LLM OUTPUT")
    else:
        logger.info(log_msg)

    write_csv(os.path.join(args.entity_dir,   f"entities_email{email_num}.csv"),
              ["id", "type", "properties"], all_ents)
    write_csv(os.path.join(args.relation_dir, f"relations_email{email_num}.csv"),
              ["start_id", "end_id", "relation", "context", "source"], all_rels)

    return n_le, n_ll

# ==============================================================================
# MERGE & STATS
# ==============================================================================

def merge_and_report(args, logger: logging.Logger) -> None:
    logger.info("Merging per-email files...")

    final_entities: Dict = {}
    for fpath in tqdm(glob.glob(os.path.join(args.entity_dir, "entities_email*.csv")), desc="Merging entities"):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for row in csv.reader(f):
                    if row and row[0] != "id":
                        final_entities[row[0]] = row
        except Exception:
            continue

    final_relations: List = []
    seen: set = set()
    for fpath in tqdm(glob.glob(os.path.join(args.relation_dir, "relations_email*.csv")), desc="Merging relations"):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for row in csv.reader(f):
                    if row and row[0] != "start_id":
                        t = tuple(row)
                        if t not in seen:
                            final_relations.append(row)
                            seen.add(t)
        except Exception:
            continue

    # Targeted Rule 2 (Appendix B.1): drop hasAuthor edges whose captured
    # evidence carries no real authorship signal, before entity resolution
    # ever sees them (see _has_hasauthor_evidence above).
    n_before = len(final_relations)
    final_relations = [
        row for row in final_relations
        if row[2] != "hasAuthor" or _has_hasauthor_evidence(row[3] if len(row) > 3 else None)
    ]
    n_dropped = n_before - len(final_relations)
    if n_dropped:
        logger.info(
            f"Targeted Rule 2: dropped {n_dropped} hasAuthor edge(s) with no "
            f"authorship-supporting evidence in their captured context."
        )

    write_csv(args.final_entities, ["id", "type", "properties"], list(final_entities.values()))
    write_csv(args.final_relations, ["start_id", "end_id", "relation", "context", "source"], final_relations)

    try:
        df_ent = pd.read_csv(args.final_entities)
        df_rel = pd.read_csv(args.final_relations)

        is_regex = (
            df_rel.iloc[:, -1].astype(str).str.contains("header", case=False, na=False) |
            df_rel.iloc[:, -2].astype(str).str.contains("Header Metadata", case=False, na=False)
        )
        regex_rels = df_rel[is_regex]
        llm_rels   = df_rel[~is_regex]
        regex_ids  = set(regex_rels.iloc[:, 0].astype(str)).union(set(regex_rels.iloc[:, 1].astype(str)))

        def classify(row):
            if row['type'] in ("Email", "MailThread"): return "Regex"
            if row['type'] == "Person": return "Regex" if str(row['id']) in regex_ids else "LLM"
            return "LLM"

        df_ent['Source'] = df_ent.apply(classify, axis=1)

        logger.info("\n" + "=" * 40)
        logger.info("EXTRACTION STATISTICS")
        logger.info("=" * 40)
        logger.info(f"\nTotal Unique Entities : {len(df_ent)}")
        logger.info(f"  - By Regex : {len(df_ent[df_ent['Source'] == 'Regex'])} (Emails, Threads, Header Persons)")
        logger.info(f"  - By LLM   : {len(df_ent[df_ent['Source'] == 'LLM'])} (Body entities)")
        logger.info(f"\nTotal Relations : {len(df_rel)}")
        logger.info(f"  - By Regex : {len(regex_rels)}")
        logger.info(f"  - By LLM   : {len(llm_rels)}")
        logger.info("\nEntity Breakdown:")
        logger.info(df_ent.groupby(['Source', 'type']).size().to_string())
        logger.info("\nRelation Breakdown:")
        if 'relation' in df_rel.columns:
            logger.info(df_rel.groupby(['relation']).size().sort_values(ascending=False).to_string())
        logger.info("=" * 40)
    except Exception as e:
        logger.warning(f"Could not compute final stats: {e}")

    logger.info(f"\nJOB COMPLETE. Entities: {len(final_entities)}, Relations: {len(final_relations)}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified KG extraction pipeline for PERK")
    parser.add_argument("--model", required=True,
                        choices=["gemma", "llama", "qwen", "qwen32b", "openai", "gptoss"],
                        help="Model backend")
    parser.add_argument("--model_path", default=None,
                        help="HuggingFace model ID or OpenAI model name (e.g. gpt-4.1). "
                             "Optional for aliases with a default (e.g. gptoss -> openai/gpt-oss-20b).")
    parser.add_argument("--input_file", required=True,
                        help="Path to preprocessed PATRA dataset")
    parser.add_argument("--output_dir", required=True,
                        help="Root output directory")
    parser.add_argument("--prompt_file", default=str(DEFAULT_PROMPT),
                        help=f"Path to system prompt file (default: {DEFAULT_PROMPT})")
    parser.add_argument("--gpu", default=None,
                        help="Physical GPU id(s) to pin for local vLLM models "
                             "(PCI-bus order; matches nvidia-smi). Use a comma list for "
                             "tensor parallelism, e.g. 0,1.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM tensor-parallel GPUs for local models (default: 1; "
                             "use >1 for large models like Qwen2.5-32B).")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="vLLM GPU memory fraction for local models (default: 0.90)")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="vLLM max context length for local models (default: 8192)")
    parser.add_argument("--base_url", default=None,
                        help="OpenAI-compatible base URL for a local server "
                             "(e.g. http://localhost:8000/v1 for vLLM serving gpt-oss-20b). "
                             "Use with --model openai or gptoss.")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max tokens to generate (default: 2048)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed emails (checks entity_extractions/ for existing files)")
    parser.add_argument("--save_debug", action="store_true", default=True,
                        help="Save raw LLM responses per email for debugging")
    args = parser.parse_args()

    # Resolve a default model name for aliases that have one (e.g. gptoss)
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATHS.get(args.model.lower())
    if not args.model_path:
        parser.error(f"--model_path is required for model '{args.model}'")

    # Derived paths
    args.entity_dir     = os.path.join(args.output_dir, "entity_extractions")
    args.relation_dir   = os.path.join(args.output_dir, "relation_extractions")
    args.final_dir      = os.path.join(args.output_dir, "final_outputs")
    args.final_entities = os.path.join(args.final_dir, "entities_final.csv")
    args.final_relations= os.path.join(args.final_dir, "relations_final.csv")

    for d in [args.output_dir, args.entity_dir, args.relation_dir, args.final_dir]:
        os.makedirs(d, exist_ok=True)

    logger = setup_logger(os.path.join(args.final_dir, "extraction_process.log"))
    logger.info("=" * 80)
    logger.info(f" STARTING KG EXTRACTION | Model: {args.model} | Path: {args.model_path}")
    logger.info("=" * 80)

    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()
    logger.info(f"Loaded prompt from: {args.prompt_file}")

    engine, sampling, client = load_model(args)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        all_emails = [e.strip() for e in f.read().split('EMAIL_END') if e.strip()]
    logger.info(f"Found {len(all_emails)} emails.")

    # Resume: collect already-processed email numbers
    processed_ids: set = set()
    if args.resume:
        for fpath in glob.glob(os.path.join(args.entity_dir, "entities_email*.csv")):
            match = re.search(r"entities_email(\d+)\.csv", os.path.basename(fpath))
            if match:
                processed_ids.add(int(match.group(1)))
        logger.info(f"Resume mode: skipping {len(processed_ids)} already-processed emails.")

    start_time = time.time()
    pbar = tqdm(all_emails, desc="Extracting", unit="email")

    for i, email_text in enumerate(pbar):
        email_num = i + 1

        if args.resume and email_num in processed_ids:
            continue

        pbar.set_description(f"Email {email_num}")
        try:
            n_ent, n_rel = process_email(
                email_num, email_text, engine, sampling, client,
                system_prompt, args, logger
            )
            pbar.set_postfix({"Ents": n_ent, "Rels": n_rel})
        except Exception as e:
            logger.error(f"CRITICAL ERROR on Email {email_num}: {e}")
            logger.exception("Full traceback:")
            continue

        if email_num % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Progress: {email_num}/{len(all_emails)} | Avg: {elapsed / email_num:.2f}s/email")

    merge_and_report(args, logger)


if __name__ == "__main__":
    main()
