# -*- coding: utf-8 -*-
import argparse
import gc
import logging
import os
import re
import sys
from pathlib import Path


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
import pandas as pd
from tqdm import tqdm


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


def build_prompt(golden_file):
    sys_p = _PROMPT_FILE.read_text(encoding="utf-8")

    df = pd.read_csv(golden_file)
    df = df[df["label"].isin(["MATCH", "NO_MATCH"])]

    few_shot = pd.concat([
        df[df["label"] == "MATCH"].sample(n=3, random_state=42),
        df[df["label"] == "NO_MATCH"].sample(n=3, random_state=42),
    ])

    for _, r in few_shot.iterrows():
        sys_p += (
            f"Entity 1: {r['entity_label_1']} | Context: {r['evidence_1']}\n"
            f"Entity 2: {r['entity_label_2']} | Context: {r['evidence_2']}\n"
            f"Output: {r['label']}\n\n"
        )
    return sys_p


def user_prompt_for(r):
    return (
        f"Entity 1: {r['entity_label_1']} | Context: {r['evidence_1']}\n"
        f"Entity 2: {r['entity_label_2']} | Context: {r['evidence_2']}\n"
        f"Output:"
    )


def parse_label(text):
    text = (text or "").strip().upper()
    return "MATCH" if "MATCH" in text and "NO_MATCH" not in text else "NO_MATCH"


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
    )
    params = SamplingParams(temperature=0.0, max_tokens=5)
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
            row_dict["llm_prediction"] = parse_label(output.outputs[0].text)
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
    """OpenAI-compatible API judging (one request per pair, no GPU needed)."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(base_url=args.base_url, api_key=api_key or "EMPTY") if args.base_url \
        else OpenAI(api_key=api_key)

    # The gpt-5 family only accepts the default temperature (1); sending any other
    # value errors. For all other models we decode deterministically (temp 0) unless
    # the user overrides via --temperature.
    is_gpt5 = bool(re.search(r"gpt-5", str(args.model), re.IGNORECASE))
    endpoint = args.base_url or "OpenAI API"
    logging.info(f"Using OpenAI-compatible endpoint: {endpoint} | model: {args.model}")
    if is_gpt5:
        logging.info("gpt-5 family detected -> using default temperature (1).")

    processed_in_chunk = 0
    batch_results = []
    for idx, r in tqdm(list(df_to_process.iterrows()), desc="Judging pairs"):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt_for(r)},
        ]
        kwargs = dict(model=args.model, messages=messages)
        if not is_gpt5:
            kwargs["temperature"] = args.temperature if args.temperature is not None else 0.0
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content
        except Exception as e:
            logging.error(f"API call failed at row {idx}: {e}")
            df_out.to_csv(args.output, index=False)
            raise

        row_dict = df.loc[idx].to_dict()
        row_dict["index"] = idx
        row_dict["llm_prediction"] = parse_label(text)
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
    parser.add_argument("--log",         default="pipeline_step2.log")
    args = parser.parse_args()

    setup_logger(args.log)
    logging.info("======== LLM ENTITY RESOLUTION STARTED ========")
    logging.info(f"Backend: {args.backend} | Model: {args.model} | Chunk size: {args.chunk_size}")

    df = pd.read_csv(args.grey_zone)
    total_rows = len(df)
    logging.info(f"Loaded {total_rows} candidate pairs.")

    # Resume logic
    if os.path.exists(args.output):
        logging.info("Existing output detected — resuming from checkpoint...")
        df_out = pd.read_csv(args.output)
        processed_indices = set(df_out["index"])
        logging.info(f"Already processed: {len(processed_indices)} rows.")
    else:
        df_out = pd.DataFrame()
        processed_indices = set()

    df_to_process = df[~df.index.isin(processed_indices)].copy()
    logging.info(f"Rows remaining: {len(df_to_process)}")

    sys_prompt = build_prompt(args.golden_set)
    logging.info("Few-shot prompt constructed.")

    if args.backend == "openai":
        df_out = run_openai(df, df_to_process, sys_prompt, df_out, total_rows, args)
    else:
        df_out = run_vllm(df, df_to_process, sys_prompt, df_out, total_rows, args)

    logging.info("======== LLM ENTITY RESOLUTION COMPLETED ========")
    logging.info(f"Total rows: {len(df_out)} | Saved to {args.output}")


if __name__ == "__main__":
    main()
