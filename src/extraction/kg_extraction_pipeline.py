# -*- coding: utf-8 -*-
"""
Unified KG extraction pipeline for PERK.
Supports Gemma, LLaMA (pipeline) and Qwen/Qwen32b (direct generate) via HuggingFace,
and OpenAI API models.
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

try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

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

PIPELINE_MODELS = {"gemma", "llama"}
DIRECT_MODELS = {"qwen", "qwen32b"}
API_MODELS = {"openai"}

HEADER_EVIDENCE = "Header Metadata"

PREFIX_MAP = {
    "Person": "pn", "Email": "e", "MailThread": "t",
    "Paper": "pa", "PaperBib": "pb", "Conference": "c",
    "Journal": "j", "Dataset": "d", "Method": "me",
    "Task": "tk", "Metric": "mt", "Meeting": "mg",
    "SubmissionID": "s", "PaperStatus": "ps",
}

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
    Load the appropriate backend.
    Returns (model_or_pipeline, tokenizer, openai_client).
    For API models, model and tokenizer are None.
    For pipeline models, model is the HF pipeline object.
    """
    model_type = args.model.lower()

    if model_type in API_MODELS:
        if not OPENAI_AVAILABLE:
            raise ImportError("pip install openai python-dotenv")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        return None, None, OpenAI(api_key=api_key)

    if not HF_AVAILABLE:
        raise ImportError("pip install torch transformers")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")

    transformers.logging.set_verbosity_error()
    print(f"Loading model: {args.model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    if model_type in PIPELINE_MODELS:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        pipe_kwargs = dict(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        if model_type == "llama":
            pipe_kwargs["repetition_penalty"] = 1.15
        print("Model loaded.")
        return hf_pipeline("text-generation", **pipe_kwargs), tokenizer, None

    elif model_type in DIRECT_MODELS:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype="auto",
            device_map="auto",
        )
        print("Model loaded.")
        return model, tokenizer, None

# ==============================================================================
# INFERENCE
# ==============================================================================

def run_inference(messages: List[Dict], model, tokenizer, client, args) -> str:
    model_type = args.model.lower()

    if model_type in API_MODELS:
        response = client.chat.completions.create(
            model=args.model_path,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content

    elif model_type in PIPELINE_MODELS:
        # model is the HF pipeline
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = model(prompt)
        return outputs[0]['generated_text']

    elif model_type in DIRECT_MODELS:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.1,
            do_sample=True,
        )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

# ==============================================================================
# REGEX HEADER PARSING
# ==============================================================================

def parse_email_person(text: str) -> List[Dict[str, str]]:
    persons = []
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
        person = {}
        if name: person['personName'] = name.replace('"', '').strip()
        if email: person['personEmail'] = email
        if affiliation: person['affiliation'] = affiliation
        if person:
            persons.append(person)
    return persons


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
    header_info = {}
    for field, pattern in [
        ('thread_id', r'Thread ID:\s*(.+)'),
        ('mail_id',   r'Mail ID:\s*(.+)'),
    ]:
        match = re.search(pattern, email_text, re.IGNORECASE)
        if match:
            header_info[field] = match.group(1).strip()

    match = re.search(r'Date:\s*(.+)', email_text, re.IGNORECASE)
    if match:
        header_info['date'] = parse_date(match.group(1).strip())

    for field, pattern in [
        ('from',    r'From:\s*(.+?)(?=\n(?:To:|Cc:|Subject:|$))'),
        ('to',      r'To:\s*(.+?)(?=\n(?:Cc:|Subject:|$))'),
        ('cc',      r'Cc:\s*(.+?)(?=\n(?:Subject:|$))'),
    ]:
        match = re.search(pattern, email_text, re.IGNORECASE | re.DOTALL)
        if match:
            header_info[field] = match.group(1).strip()

    match = re.search(r'Subject:\s*(.+)', email_text, re.IGNORECASE)
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
    field_map = {
        "Paper": "paperTitle", "PaperBib": "doi", "Dataset": "datasetName",
        "Method": "methodName", "Task": "taskName", "Metric": "metricName",
        "Email": "mailNum", "MailThread": "threadID", "SubmissionID": "identifier",
        "Conference": "confTitle", "Journal": "journalTitle",
        "PaperStatus": "statusType", "Meeting": "meetAgenda",
    }
    if entity_type in field_map:
        return f"{entity_type.lower()}_{properties.get(field_map[entity_type], '').lower().strip()}"
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


def extract_header_entities(header_info: dict, email_num: int) -> Tuple[List, List]:
    entities, relations = [], []

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
            for person in parse_email_person(header_info[field]):
                person_id = create_unique_entity_id("Person", person)
                entities.append([person_id, "Person", json.dumps(person)])
                if 'mail_id' in header_info:
                    rel = "sentBy" if field == 'from' else "receivedBy"
                    relations.append([email_id, person_id, rel, HEADER_EVIDENCE, "header"])

    return entities, relations


def build_header_persons_context(header_entities: List) -> str:
    """Build the prompt context string listing persons found in headers."""
    persons = [e for e in header_entities if e[1] == 'Person']
    if not persons:
        return "No persons in headers."
    context = "Persons already extracted from email headers (reuse these IDs):\n"
    for eid, _, _ in persons:
        if eid in entity_registry:
            context += f"- {eid}: {json.dumps(entity_registry[eid]['properties'])}\n"
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

def process_email(email_num: int, email_text: str, model, tokenizer, client,
                  system_prompt: str, args, logger: logging.Logger) -> Tuple[int, int]:
    header_info = extract_header_info(email_text)
    body_text = extract_body(email_text)
    header_entities, header_relations = extract_header_entities(header_info, email_num)
    header_context = build_header_persons_context(header_entities)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{header_context}\n\nBody:\n{body_text}\n\nExtract JSON."}
    ]

    try:
        response_text = run_inference(messages, model, tokenizer, client, args)
    except Exception as e:
        logger.error(f"Inference error on Email {email_num}: {e}")
        response_text = "{}"

    if args.save_debug:
        with open(os.path.join(args.output_dir, f"debug_{args.model}_email{email_num}.txt"), 'w', encoding='utf-8') as f:
            f.write(response_text)

    entities_raw, relations_raw = parse_llm_output(response_text)

    body_entities: List = []
    body_relations: List = []
    temp_to_global: Dict = {}

    for temp_id, etype, props_str in entities_raw:
        try:
            props = json.loads(props_str)
            global_id = create_unique_entity_id(etype, props)
            temp_to_global[temp_id] = global_id
            row = [global_id, etype, props_str]
            if global_id not in {e[0] for e in entities_all}: entities_all.append(row)
            if global_id not in {e[0] for e in body_entities}: body_entities.append(row)
        except Exception:
            continue

    current_email_id = next((e[0] for e in header_entities if e[1] == "Email"), "unknown")

    for s_id, o_id, rel, ctx in relations_raw:
        s_glob = temp_to_global.get(s_id, s_id)
        o_glob = temp_to_global.get(o_id, o_id)
        valid_start = s_glob in entity_registry or s_glob in {e[0] for e in body_entities}
        valid_end   = o_glob in entity_registry or o_glob in {e[0] for e in body_entities}
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
                        choices=["gemma", "llama", "qwen", "qwen32b", "openai"],
                        help="Model backend")
    parser.add_argument("--model_path", required=True,
                        help="HuggingFace model ID or OpenAI model name (e.g. gpt-4.1)")
    parser.add_argument("--input_file", required=True,
                        help="Path to preprocessed PATRA dataset")
    parser.add_argument("--output_dir", required=True,
                        help="Root output directory")
    parser.add_argument("--prompt_file", default=str(DEFAULT_PROMPT),
                        help=f"Path to system prompt file (default: {DEFAULT_PROMPT})")
    parser.add_argument("--gpu", default=None,
                        help="GPU ID to use (e.g. 0 or 1)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max tokens to generate (default: 2048)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed emails (checks entity_extractions/ for existing files)")
    parser.add_argument("--save_debug", action="store_true", default=True,
                        help="Save raw LLM responses per email for debugging")
    args = parser.parse_args()

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

    model, tokenizer, client = load_model(args)

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
                email_num, email_text, model, tokenizer, client,
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
