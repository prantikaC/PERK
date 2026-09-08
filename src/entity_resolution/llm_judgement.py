# -*- coding: utf-8 -*-
import argparse
import gc
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# A bare .env file does nothing on its own -- os.environ.get() only sees
# variables actually exported into the process environment. Load it
# explicitly here so --backend openai works regardless of shell state.
# Resolved relative to this file (PERK_v2/.env), not the cwd, so it's found
# no matter what directory the script is invoked from. override=True: a
# stale key from an old `export OPENAI_API_KEY=...` left over in the shell
# session would otherwise silently shadow a corrected .env value forever --
# confirmed this actually happened (the .env key was updated but the API
# still rejected an old, different key that was still exported in the
# shell). .env is meant to be the single source of truth here, so it always
# wins over whatever the shell happens to already have set.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=True)


# --- Pin the GPU BEFORE importing torch / vLLM ----------------------------- #
# torch and vLLM read CUDA_VISIBLE_DEVICES exactly once, at import time; setting
# it afterwards has no effect. We parse --gpu from argv here, mask all other
# GPUs, and force PCI-bus ordering so --gpu N is PHYSICAL GPU N (the same number
# nvidia-smi shows). After masking, the chosen card is the only visible device,
# so vLLM (tensor_parallel_size=1) loads onto it and no other GPU can be touched.
# (Only relevant for --backend vllm; harmless for --backend openai.)
def _pin_gpu_from_argv(default="0"):
    gpu = os.environ.get("PERK_GPU", default)
    for i, a in enumerate(sys.argv):
        if a == "--gpu" and i + 1 < len(sys.argv):
            gpu = sys.argv[i + 1]
        elif a.startswith("--gpu="):
            gpu = a.split("=", 1)[1]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return str(gpu)

_PINNED_GPU = _pin_gpu_from_argv()

# torch / transformers / vllm are imported lazily inside run_vllm() so the
# OpenAI backend works on a machine without a GPU (or without vLLM installed).
import json

import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# STRICT-CHECK RE-APPLICATION (golden/candidate sets only)
# ==============================================================================
# A golden/candidate set built for calibration or evaluation is a snapshot of
# whatever FAISS proposed -- it doesn't automatically reflect strict-check
# changes made to faiss_blocking.py afterward (date mismatch, Person/Team
# EmailID ownership mismatch, Conference year mismatch, Team affiliation
# mismatch). Re-running the LLM on a pair the real pipeline would now
# auto-reject before it ever reached the LLM wastes a call on a foregone
# conclusion. These are deliberately duplicated from faiss_blocking.py
# (not imported) -- that module pins a GPU and imports torch/faiss/
# sentence_transformers at import time as a side effect, which this script
# has no reason to trigger just to reuse a few pure-Python predicates.

TITLE_YEAR_RE = re.compile(r'\d{2,4}')
ISO_YEAR_RE = re.compile(r'^\s*(\d{4})')


def extract_years_from_title(text):
    years = set()
    for m in TITLE_YEAR_RE.finditer(text or ''):
        s = m.group()
        if len(s) == 4 and s[:2] in ('19', '20'):
            years.add(int(s))
        elif len(s) == 2:
            years.add(2000 + int(s))
    return years


def extract_year_from_date(text):
    if not text:
        return set()
    m = ISO_YEAR_RE.match(text)
    if m and m.group(1)[:2] in ('19', '20'):
        return {int(m.group(1))}
    return set()


def passes_date_check(props1, props2):
    date_keys = {k for k in set(props1) | set(props2) if k.lower().endswith('date')}
    for key in date_keys:
        v1, v2 = props1.get(key), props2.get(key)
        if v1 and v2 and v1 != v2:
            return False
    return True


def passes_conference_year_check(props1, props2):
    def years_of(props):
        return extract_years_from_title(props.get('confTitle', '')) | extract_year_from_date(props.get('confDate'))
    y1, y2 = years_of(props1), years_of(props2)
    if y1 and y2:
        return bool(y1 & y2)
    if y1 or y2:
        return False
    return True


def passes_overlap_check(id1, id2, id_sets):
    s1, s2 = id_sets.get(id1, set()), id_sets.get(id2, set())
    if s1 and s2 and not (s1 & s2):
        return False
    return True


def load_strict_check_context(raw_entities_path, raw_relations_path):
    """Builds everything needed to re-run the FAISS strict checks: each
    entity's own properties/type, Person/Team -> EmailID ownership (via
    hasOwner), and Team -> Organization/Journal/Conference affiliation."""
    df_ent = pd.read_csv(raw_entities_path)
    df_rel = pd.read_csv(raw_relations_path)

    props_by_id, type_by_id = {}, {}
    for _, row in df_ent.iterrows():
        eid = str(row['id']).strip()
        type_by_id[eid] = row['type']
        try:
            props_by_id[eid] = json.loads(row['properties'])
        except Exception:
            props_by_id[eid] = {}

    owner_email_ids = defaultdict(set)
    team_affiliations = defaultdict(set)
    for _, r in df_rel.iterrows():
        if r['relation'] == 'hasOwner':
            owner_email_ids[r['end_id']].add(r['start_id'])
        elif r['relation'] == 'affiliation':
            team_affiliations[r['start_id']].add(r['end_id'])

    return props_by_id, type_by_id, owner_email_ids, team_affiliations


def strict_check_verdict(id1, id2, ctx):
    """Returns False if the real pipeline would auto-reject this pair before
    it ever reached the LLM (given the CURRENT strict checks); True if it
    would still reach the LLM (checks passed, or entities/ids unknown --
    unknown data was never grounds to auto-reject in these checks anyway)."""
    props_by_id, type_by_id, owner_email_ids, team_affiliations = ctx
    id1, id2 = str(id1), str(id2)
    if id1 not in props_by_id or id2 not in props_by_id:
        return True
    p1, p2 = props_by_id[id1], props_by_id[id2]
    if not passes_date_check(p1, p2):
        return False
    e_type = type_by_id.get(id1)
    if e_type in ('Person', 'Team') and not passes_overlap_check(id1, id2, owner_email_ids):
        return False
    if e_type == 'Conference' and not passes_conference_year_check(p1, p2):
        return False
    if e_type == 'Team' and not passes_overlap_check(id1, id2, team_affiliations):
        return False
    return True


def setup_logger(log_file):
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )


_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "llm_judgement_prompt.txt"


FEW_SHOT_KEY_COLS = ['entity_label_1', 'evidence_1', 'entity_label_2', 'evidence_2']


def _row_key(row):
    return tuple(str(row[c]).strip() for c in FEW_SHOT_KEY_COLS)


def build_prompt():
    # This system prompt is sent in FULL on EVERY grey-zone row -- with tens
    # of thousands of rows, its size is the dominant cost driver, not the
    # per-row evidence text (confirmed empirically: shrinking evidence from
    # ~2000 to ~130 chars/row barely moved total cost). No few-shot examples
    # baked in -- just the bare MATCH/NO_MATCH + confidence instructions
    # already in the prompt file, paid for once per request instead of once
    # per request PLUS 6 embedded example pairs.
    return _PROMPT_FILE.read_text(encoding="utf-8")


def build_golden_label_lookup(golden_file):
    """key -> human-annotated label, for EVERY row in the golden set (not
    just the handful sampled as few-shot demonstrations above). The golden
    set was originally sampled FROM a production grey_zone.csv run over
    this same entity pool -- so when the full pipeline runs again (a fresh
    grey_zone.csv over all entities, not just the golden sample), the exact
    same candidate pair can legitimately reappear there too. Without this
    lookup, that pair would go back to the LLM and get a SECOND, independent
    decision -- wasting a call, and risking an inconsistent verdict, on a
    pair that already has an authoritative, human-given answer. Whenever a
    pair being processed matches one already in the golden set, that label
    is used directly instead of asking the LLM again."""
    df = pd.read_csv(golden_file)
    df = df[df["label"].isin(["MATCH", "NO_MATCH"])]
    return {_row_key(r): r['label'] for _, r in df.iterrows()}


def user_prompt_for(r):
    return (
        f"Entity 1: {r['entity_label_1']} | Context: {r['evidence_1']}\n"
        f"Entity 2: {r['entity_label_2']} | Context: {r['evidence_2']}\n"
        f"Output:"
    )


CONFIDENCE_RE = re.compile(r'(\d*\.?\d+)')


def parse_label_and_confidence(text):
    """Expects 'MATCH <confidence>' / 'NO_MATCH <confidence>'. Falls back to
    confidence 0.5 (maximally uncertain, neither corroborating nor
    contradicting a FAISS-only signal) if the model didn't include a usable
    number -- this can still happen despite the prompt instruction, and a
    missing confidence shouldn't silently masquerade as a confident one."""
    raw = (text or "").strip().upper()
    label = "MATCH" if "MATCH" in raw and "NO_MATCH" not in raw else "NO_MATCH"

    after_label = raw.split("MATCH", 1)[-1] if "MATCH" in raw else raw
    m = CONFIDENCE_RE.search(after_label)
    if m:
        try:
            confidence = max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            confidence = 0.5
    else:
        confidence = 0.5
    return label, confidence


def run_vllm(df, df_to_process, sys_prompt, df_out, total_rows, args):
    """Local Qwen-style judging via vLLM (batched, GPU)."""
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    logging.info(f"Pinned to physical GPU {_PINNED_GPU} (visible as cuda:0)")
    logging.info("Initializing vLLM engine...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=4096,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=12)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    logging.info("Beginning inference...")
    for start in tqdm(range(0, len(df_to_process), args.chunk_size), desc="Processing batches"):
        chunk = df_to_process.iloc[start:start + args.chunk_size]
        prompts, valid_indices = [], []

        for idx, r in chunk.iterrows():
            full_prompt = (
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt_for(r)}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            tokens = tokenizer(full_prompt)["input_ids"]
            if len(tokens) > 3800:
                full_prompt = tokenizer.decode(tokens[:3800])
            prompts.append(full_prompt)
            valid_indices.append(idx)

        if not prompts:
            continue

        try:
            outputs = llm.generate(prompts, params, use_tqdm=False)
        except Exception as e:
            logging.error(f"Generation crashed at batch {start}: {e}")
            df_out.to_csv(args.output, index=False)
            raise

        batch_results = []
        for idx, output in zip(valid_indices, outputs):
            row_dict = df.loc[idx].to_dict()
            row_dict["index"] = idx
            row_dict["llm_prediction"], row_dict["llm_confidence"] = parse_label_and_confidence(
                output.outputs[0].text
            )
            batch_results.append(row_dict)

        if batch_results:
            df_out = pd.concat([df_out, pd.DataFrame(batch_results)], ignore_index=True)

        df_out.to_csv(args.output, index=False)
        logging.info(f"Checkpoint saved. {len(df_out)} / {total_rows} rows complete.")

        del prompts, outputs
        gc.collect()
        torch.cuda.empty_cache()

    return df_out


def run_openai(df, df_to_process, sys_prompt, df_out, total_rows, args):
    """OpenAI-compatible API judging. Requests run concurrently (--concurrency
    worker threads) since each call is pure I/O wait -- at 1 request per pair
    sequentially, tens of thousands of grey-zone candidates take hours; a
    thread pool cuts that roughly by the concurrency factor without touching
    precision/recall at all."""
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(base_url=args.base_url, api_key=api_key or "EMPTY") if args.base_url \
        else OpenAI(api_key=api_key)

    # The gpt-5 family only accepts the default temperature (1); sending any other
    # value errors. For all other models we decode deterministically (temp 0) unless
    # the user overrides via --temperature.
    is_gpt5 = bool(re.search(r"gpt-5", str(args.model), re.IGNORECASE))
    endpoint = args.base_url or "OpenAI API"
    logging.info(f"Using OpenAI-compatible endpoint: {endpoint} | model: {args.model} | "
                 f"concurrency: {args.concurrency}")
    if is_gpt5:
        logging.info("gpt-5 family detected -> using default temperature (1).")

    def judge_one(idx, r):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt_for(r)},
        ]
        kwargs = dict(model=args.model, messages=messages)
        if not is_gpt5:
            kwargs["temperature"] = args.temperature if args.temperature is not None else 0.0
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    processed_in_chunk = 0
    batch_results = []
    rows = list(df_to_process.iterrows())

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(judge_one, idx, r): idx for idx, r in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging pairs"):
            idx = futures[future]
            try:
                text = future.result()
            except Exception as e:
                logging.error(f"API call failed at row {idx}: {e} -- skipping; "
                              f"rerun to pick it up via checkpoint resume.")
                continue

            row_dict = df.loc[idx].to_dict()
            row_dict["index"] = idx
            row_dict["llm_prediction"], row_dict["llm_confidence"] = parse_label_and_confidence(text)
            batch_results.append(row_dict)
            processed_in_chunk += 1

            if processed_in_chunk >= args.chunk_size:
                df_out = pd.concat([df_out, pd.DataFrame(batch_results)], ignore_index=True)
                df_out.to_csv(args.output, index=False)
                logging.info(f"Checkpoint saved. {len(df_out)} / {total_rows} rows complete.")
                batch_results, processed_in_chunk = [], 0

    if batch_results:
        df_out = pd.concat([df_out, pd.DataFrame(batch_results)], ignore_index=True)
        df_out.to_csv(args.output, index=False)
        logging.info(f"Checkpoint saved. {len(df_out)} / {total_rows} rows complete.")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="LLM Entity Resolution (vLLM or OpenAI backend)")
    parser.add_argument("--grey_zone",   required=True, help="Grey zone CSV from FAISS blocking")
    parser.add_argument("--golden_set",  required=True, help="Annotated golden set CSV for few-shot prompt")
    parser.add_argument("--output",      required=True, help="Output resolved CSV")
    parser.add_argument("--backend",     choices=["vllm", "openai"], default="vllm",
                        help="Judging backend: 'vllm' (local Qwen, default) or 'openai' (API).")
    parser.add_argument("--model",       default="Qwen/Qwen2.5-32B-Instruct",
                        help="HuggingFace model id for vLLM, or OpenAI model name "
                             "(e.g. gpt-5.1) for --backend openai.")
    parser.add_argument("--concurrency", type=int, default=15,
                        help="Concurrent worker threads for --backend openai (default: 15). "
                             "Each request is pure I/O wait, so this scales throughput roughly "
                             "linearly -- lower it if you hit OpenAI rate limits (429s).")
    parser.add_argument("--base_url",    default=None,
                        help="OpenAI-compatible base URL (e.g. a local vLLM/Ollama server). "
                             "Only used with --backend openai; omit for the real OpenAI API.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature for --backend openai (non-gpt-5 models). "
                             "Default deterministic (0.0). Ignored for the gpt-5 family.")
    parser.add_argument("--gpu",         default="0",
                        help="Physical GPU id to pin for --backend vllm (PCI-bus order; "
                             "matches nvidia-smi). The chosen card is the only one made visible. "
                             "Default: 0")
    parser.add_argument("--chunk_size",  type=int, default=48,
                        help="Rows per checkpoint (vLLM batch size / OpenAI save interval).")
    parser.add_argument("--max_num_seqs",type=int, default=12, help="vLLM max concurrent sequences.")
    parser.add_argument("--raw_entities", default=None,
                        help="Optional raw entities CSV (id,type,properties). When given together "
                             "with --raw_relations, pairs the CURRENT FAISS strict checks (date, "
                             "Person/Team EmailID ownership, Conference year, Team affiliation) "
                             "would now auto-reject before ever reaching the LLM are skipped here "
                             "too and auto-labeled NO_MATCH, instead of spending a call on a "
                             "foregone conclusion. Needs entity1_id/entity2_id columns in the "
                             "input CSV (present in grey_zone.csv and the golden/candidate sets).")
    parser.add_argument("--raw_relations", default=None,
                        help="Optional raw relations CSV, paired with --raw_entities.")
    parser.add_argument("--log",         default="pipeline_step2.log")
    args = parser.parse_args()

    setup_logger(args.log)
    logging.info("======== LLM ENTITY RESOLUTION STARTED ========")
    logging.info(f"Backend: {args.backend} | Model: {args.model} | Chunk size: {args.chunk_size}")

    df = pd.read_csv(args.grey_zone)
    total_rows = len(df)
    logging.info(f"Loaded {total_rows} candidate pairs.")

    # Resume logic. A stale, empty/truncated output file (e.g. left over from
    # a crashed prior attempt before any real checkpoint was written) isn't a
    # valid checkpoint -- treat it as if no output existed yet rather than
    # crashing on an unparseable file.
    df_out = pd.DataFrame()
    processed_indices = set()
    if os.path.exists(args.output):
        try:
            df_out = pd.read_csv(args.output)
            if df_out.empty or "index" not in df_out.columns:
                raise ValueError("existing output has no usable rows/columns")
            processed_indices = set(df_out["index"])
            logging.info(f"Existing output detected — resuming from checkpoint. "
                         f"Already processed: {len(processed_indices)} rows.")
        except (pd.errors.EmptyDataError, ValueError) as e:
            logging.info(f"Existing output at {args.output} isn't a usable checkpoint ({e}) -- "
                         f"starting fresh instead of resuming from it.")
            df_out = pd.DataFrame()
            processed_indices = set()

    df_to_process = df[~df.index.isin(processed_indices)].copy()
    logging.info(f"Rows remaining: {len(df_to_process)}")

    sys_prompt = build_prompt()
    logging.info("Few-shot prompt constructed.")

    # Skip entirely when --grey_zone IS the golden set (a calibration/
    # evaluation run judging the golden set on purpose, to measure the LLM's
    # own accuracy/confidence against it) -- reusing the label there would
    # short-circuit every row to its own ground truth and the LLM would
    # never actually be asked to judge anything, defeating the run's purpose.
    is_self_evaluation = os.path.abspath(args.grey_zone) == os.path.abspath(args.golden_set)
    golden_lookup = {} if is_self_evaluation else build_golden_label_lookup(args.golden_set)
    if is_self_evaluation:
        logging.info("--grey_zone is the golden set itself -- judging every row for real "
                      "(golden-label reuse is skipped so this run actually measures the LLM).")
    if golden_lookup and len(df_to_process):
        golden_key = df_to_process.apply(_row_key, axis=1)
        is_known = golden_key.isin(golden_lookup)
        n_known = int(is_known.sum())
        if n_known:
            logging.info(f"{n_known} pairs already have a human-annotated ground-truth label in "
                         f"the golden set (it was originally sampled from a grey_zone.csv over "
                         f"this same entity pool, so the same pair can legitimately reappear here "
                         f"too) -- reusing that label directly instead of asking the LLM to "
                         f"re-decide an already-answered pair.")
            known_rows = df_to_process[is_known].copy()
            known_rows['index'] = known_rows.index
            known_rows['llm_prediction'] = golden_key[is_known].map(golden_lookup)
            known_rows['llm_confidence'] = 1.0
            df_out = pd.concat([df_out, known_rows], ignore_index=True)
            df_out.to_csv(args.output, index=False)
            df_to_process = df_to_process[~is_known].copy()

    if args.raw_entities and args.raw_relations and len(df_to_process):
        ctx = load_strict_check_context(args.raw_entities, args.raw_relations)
        would_reject = df_to_process.apply(
            lambda r: not strict_check_verdict(r['entity1_id'], r['entity2_id'], ctx), axis=1
        )
        n_rejected = int(would_reject.sum())
        if n_rejected:
            logging.info(f"Auto-labeling {n_rejected} pairs NO_MATCH without an LLM call -- "
                         f"the current FAISS strict checks (date / Person-Team EmailID ownership "
                         f"/ Conference year / Team affiliation) would reject them before they "
                         f"ever reached the LLM in the real pipeline.")
            rejected_rows = df_to_process[would_reject].copy()
            rejected_rows['index'] = rejected_rows.index
            rejected_rows['llm_prediction'] = 'NO_MATCH'
            rejected_rows['llm_confidence'] = 1.0
            df_out = pd.concat([df_out, rejected_rows], ignore_index=True)
            df_out.to_csv(args.output, index=False)
            df_to_process = df_to_process[~would_reject].copy()

    if args.backend == "openai":
        df_out = run_openai(df, df_to_process, sys_prompt, df_out, total_rows, args)
    else:
        df_out = run_vllm(df, df_to_process, sys_prompt, df_out, total_rows, args)

    logging.info("======== LLM ENTITY RESOLUTION COMPLETED ========")
    logging.info(f"Total rows: {len(df_out)} | Saved to {args.output}")


if __name__ == "__main__":
    main()
