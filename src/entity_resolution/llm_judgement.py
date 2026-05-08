# -*- coding: utf-8 -*-
import argparse
import gc
import logging
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


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


def build_prompt(golden_file):
    df = pd.read_csv(golden_file)
    df = df[df["label"].isin(["MATCH", "NO_MATCH"])]

    few_shot = pd.concat([
        df[df["label"] == "MATCH"].sample(n=3, random_state=42),
        df[df["label"] == "NO_MATCH"].sample(n=3, random_state=42),
    ])

    sys_p = (
        "You are an expert ontology curator. Determine if Entity 1 and Entity 2 "
        "refer to the exact same real-world concept.\n"
        "RULES:\n"
        "1. Semantic equivalence matters more than exact spelling.\n"
        "2. Acronyms matching full expansions are a MATCH.\n"
        "3. STRICT REJECTION: Contradicting dates or emails means NO_MATCH.\n"
        "4. Sub-components are NO_MATCH.\n"
        "Respond ONLY with 'MATCH' or 'NO_MATCH'.\n\nEXAMPLES:\n"
    )
    for _, r in few_shot.iterrows():
        sys_p += (
            f"Entity 1: {r['entity_label_1']} | Context: {r['evidence_1']}\n"
            f"Entity 2: {r['entity_label_2']} | Context: {r['evidence_2']}\n"
            f"Output: {r['label']}\n\n"
        )
    return sys_p


def main():
    parser = argparse.ArgumentParser(description="vLLM Entity Resolution")
    parser.add_argument("--grey_zone",   required=True, help="Grey zone CSV from FAISS blocking")
    parser.add_argument("--golden_set",  required=True, help="Annotated golden set CSV for few-shot prompt")
    parser.add_argument("--output",      required=True, help="Output resolved CSV")
    parser.add_argument("--model",       default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--chunk_size",  type=int, default=48)
    parser.add_argument("--max_num_seqs",type=int, default=12)
    parser.add_argument("--log",         default="pipeline_step2.log")
    args = parser.parse_args()

    setup_logger(args.log)
    logging.info("======== LLM ENTITY RESOLUTION STARTED ========")
    logging.info(f"Model: {args.model} | Chunk size: {args.chunk_size}")

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
            user_prompt = (
                f"Entity 1: {r['entity_label_1']} | Context: {r['evidence_1']}\n"
                f"Entity 2: {r['entity_label_2']} | Context: {r['evidence_2']}\n"
                f"Output:"
            )
            full_prompt = (
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
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
            text = output.outputs[0].text.strip().upper()
            pred = "MATCH" if "MATCH" in text and "NO_MATCH" not in text else "NO_MATCH"
            row_dict = df.loc[idx].to_dict()
            row_dict["index"] = idx
            row_dict["llm_prediction"] = pred
            batch_results.append(row_dict)

        if batch_results:
            df_out = pd.concat([df_out, pd.DataFrame(batch_results)], ignore_index=True)

        df_out.to_csv(args.output, index=False)
        logging.info(f"Checkpoint saved. {len(df_out)} / {total_rows} rows complete.")

        del prompts, outputs
        gc.collect()
        torch.cuda.empty_cache()

    logging.info("======== LLM ENTITY RESOLUTION COMPLETED ========")
    logging.info(f"Total rows: {len(df_out)} | Saved to {args.output}")


if __name__ == "__main__":
    main()
