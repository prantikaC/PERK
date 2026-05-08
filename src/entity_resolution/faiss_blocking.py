# -*- coding: utf-8 -*-
import argparse
import json
import logging

import faiss
import pandas as pd
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer

TARGET_KEYS = ['datasetName', 'methodName', 'taskName', 'metricName',
               'personName', 'journalTitle', 'confTitle']
STRICT_KEYS = ['taskDate', 'personEmail']


def passes_strict_metadata_check(props1, props2):
    for key in STRICT_KEYS:
        if props1.get(key) != props2.get(key):
            return False
    return True


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


def main():
    parser = argparse.ArgumentParser(description="FAISS Blocking Funnel for Entity Resolution")
    parser.add_argument("--entities",   required=True, help="Input entities CSV")
    parser.add_argument("--relations",  required=True, help="Input relations CSV")
    parser.add_argument("--output",     required=True, help="Output grey zone CSV")
    parser.add_argument("--threshold",  type=float, required=True,
                        help="Calibrated similarity floor (e.g. 0.6547)")
    parser.add_argument("--top_k",      type=int, default=10, help="FAISS neighbourhood span (default: 10)")
    parser.add_argument("--model",      default="all-mpnet-base-v2", help="SentenceTransformer model name")
    parser.add_argument("--log",        default="pipeline_step1.log")
    args = parser.parse_args()

    setup_logger(args.log)
    logging.info(f"Starting FAISS Funnel | threshold={args.threshold} | top_k={args.top_k}")

    df_ent = pd.read_csv(args.entities)
    df_rel = pd.read_csv(args.relations)

    df_rel['start_id'] = df_rel['start_id'].astype(str).str.strip()
    df_rel['end_id']   = df_rel['end_id'].astype(str).str.strip()
    valid_ids = set(df_rel['start_id']).union(set(df_rel['end_id']))

    parsed = []
    for _, row in df_ent.iterrows():
        eid = str(row['id']).strip()
        if eid not in valid_ids:
            continue
        try:
            props = json.loads(row['properties'])
        except Exception:
            continue
        for k in TARGET_KEYS:
            if k in props:
                parsed.append({"id": eid, "type": row['type'], "label": str(props[k]), "raw_props": props})
                break

    df_parsed = pd.DataFrame(parsed)
    logging.info(f"Retained {len(df_parsed)} connected entities matching target ontology.")

    ctx_map = defaultdict(list)
    for _, r in df_rel.iterrows():
        ctx = str(r['context']).strip()
        if ctx and ctx.lower() != "nan":
            ctx_map[r['start_id']].append(ctx)
            ctx_map[r['end_id']].append(ctx)

    df_parsed['context'] = df_parsed['id'].apply(
        lambda x: " | ".join(ctx_map.get(x, ["Missing"]))
    )
    df_parsed['text_to_embed'] = df_parsed.apply(
        lambda x: f"Entity: {x['label']}. Evidence: {x['context']}", axis=1
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading embedding model '{args.model}' to {device}...")
    model = SentenceTransformer(args.model, device=device)

    all_pairs, seen = [], set()
    strict_rejections = threshold_rejections = 0

    for e_type in df_parsed['type'].unique():
        df_t = df_parsed[df_parsed['type'] == e_type].reset_index(drop=True)
        if len(df_t) < 2:
            continue

        embeddings = model.encode(
            df_t['text_to_embed'].tolist(),
            batch_size=64, convert_to_numpy=True, normalize_embeddings=True
        )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        sims, idxs = index.search(embeddings, min(args.top_k, len(df_t)))

        for i in range(len(df_t)):
            for rank in range(1, min(args.top_k, len(df_t))):
                sim = float(sims[i][rank])
                if sim <= args.threshold:
                    threshold_rejections += 1
                    break

                j = idxs[i][rank]
                if i == j:
                    continue

                p_id = tuple(sorted([df_t.iloc[i]['id'], df_t.iloc[j]['id']]))
                if p_id in seen:
                    continue

                if not passes_strict_metadata_check(df_t.iloc[i]['raw_props'], df_t.iloc[j]['raw_props']):
                    strict_rejections += 1
                    continue

                seen.add(p_id)
                all_pairs.append({
                    "entity_type":    e_type,
                    "similarity_score": sim,
                    "entity1_id":     df_t.iloc[i]['id'],
                    "entity_label_1": df_t.iloc[i]['label'],
                    "evidence_1":     df_t.iloc[i]['context'],
                    "entity2_id":     df_t.iloc[j]['id'],
                    "entity_label_2": df_t.iloc[j]['label'],
                    "evidence_2":     df_t.iloc[j]['context'],
                })

    df_grey = pd.DataFrame(all_pairs)
    df_grey.to_csv(args.output, index=False)

    logging.info("--- FAISS Funnel Results ---")
    logging.info(f"Strict metadata rejections : {strict_rejections}")
    logging.info(f"Threshold rejections       : {threshold_rejections} (estimated)")
    logging.info(f"Grey zone candidates saved : {len(df_grey)}")
    logging.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
