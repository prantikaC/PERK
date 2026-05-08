# -*- coding: utf-8 -*-
"""
Generate candidate entity pairs for manual annotation.
Uses FAISS semantic blocking with strict metadata filtering and
a lexical gate for proper-noun entity types (Person, Conference, Journal).
Output is a CSV for annotation (label column left blank).

Usage:
    python generate_golden_candidates.py \
        --entities entities_final.csv \
        --relations relations_final.csv \
        --output annotation_candidates.csv \
        --top_k 10 \
        --semantic_floor 0.65 \
        --max_pairs 3
"""

import argparse
import json
import re
from collections import defaultdict

import faiss
import jellyfish
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

TARGET_KEYS = ['datasetName', 'methodName', 'taskName', 'metricName',
               'personName', 'journalTitle', 'confTitle']
STRICT_KEYS = ['taskDate', 'personEmail']
PROPER_NOUN_TYPES = ['Person', 'Conference', 'Journal']


def passes_strict_metadata_check(props1, props2):
    for key in STRICT_KEYS:
        if props1.get(key) != props2.get(key):
            return False
    return True


def load_and_preprocess(ent_path, rel_path, model, device):
    df_ent = pd.read_csv(ent_path)
    df_rel = pd.read_csv(rel_path)

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
        for key in TARGET_KEYS:
            if key in props:
                parsed.append({"id": eid, "type": row['type'],
                                "label": str(props[key]), "raw_props": props})
                break

    df_entities = pd.DataFrame(parsed)
    print(f"Retained {len(df_entities)} connected entities.")

    ctx_map = defaultdict(list)
    for _, row in df_rel.iterrows():
        ctx = str(row['context']).strip()
        if not ctx or ctx.lower() == "nan":
            continue
        ctx_map[row['start_id']].append(ctx)
        ctx_map[row['end_id']].append(ctx)

    df_entities['context'] = df_entities['id'].apply(
        lambda x: " | ".join(ctx_map.get(x, ["Context missing"]))
    )
    df_entities['text_to_embed'] = df_entities.apply(
        lambda x: f"Entity: {x['label']}. Evidence: {x['context']}", axis=1
    )
    return df_entities


def generate_candidate_pairs(df_entities, model, device, top_k, semantic_floor,
                              semantic_floor_proper, max_pairs):
    all_pairs, seen = [], set()

    for e_type in df_entities['type'].unique():
        df_type = df_entities[df_entities['type'] == e_type].reset_index(drop=True)
        if len(df_type) < 2:
            continue

        print(f"Processing {e_type} ({len(df_type)} entities)...")
        texts = df_type['text_to_embed'].tolist()
        embeddings = model.encode(
            texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True
        )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        sims, indices = index.search(embeddings, min(top_k, len(df_type)))
        floor = semantic_floor_proper if e_type in PROPER_NOUN_TYPES else semantic_floor

        for i in range(len(df_type)):
            pairs_added = 0
            for rank in range(1, min(top_k, len(df_type))):
                j = int(indices[i][rank])
                sim = float(sims[i][rank])
                if i == j or sim < floor:
                    continue

                pair_ids = tuple(sorted([df_type.iloc[i]['id'], df_type.iloc[j]['id']]))
                if pair_ids in seen:
                    continue

                if not passes_strict_metadata_check(df_type.iloc[i]['raw_props'],
                                                    df_type.iloc[j]['raw_props']):
                    continue

                # Lexical gate for proper nouns
                if e_type in PROPER_NOUN_TYPES:
                    jw = jellyfish.jaro_winkler_similarity(
                        df_type.iloc[i]['label'], df_type.iloc[j]['label']
                    )
                    if jw < 0.92:
                        continue

                seen.add(pair_ids)
                all_pairs.append({
                    "entity_type":    e_type,
                    "similarity_score": sim,
                    "entity1_id":     df_type.iloc[i]['id'],
                    "entity_label_1": df_type.iloc[i]['label'],
                    "evidence_1":     df_type.iloc[i]['context'],
                    "entity2_id":     df_type.iloc[j]['id'],
                    "entity_label_2": df_type.iloc[j]['label'],
                    "evidence_2":     df_type.iloc[j]['context'],
                    "label":          "",
                })
                pairs_added += 1
                if pairs_added >= max_pairs:
                    break

    return pd.DataFrame(all_pairs)


def adaptive_sample(df_pairs):
    n = len(df_pairs)
    if n == 0:
        return df_pairs
    if n < 1000:
        target = int(0.4 * n)
    elif n < 5000:
        target = min(500, int(0.15 * n))
    else:
        target = 500

    print(f"Adaptive annotation target: {target} pairs (from {n} candidates).")
    if n <= target:
        return df_pairs

    sampled = (
        df_pairs.groupby('entity_type', group_keys=False)
        .apply(lambda x: x.sample(n=max(1, int(target * len(x) / n)), random_state=42))
    )
    if len(sampled) > target:
        sampled = sampled.sample(n=target, random_state=42)
    elif len(sampled) < target:
        remaining = df_pairs.drop(sampled.index)
        needed = target - len(sampled)
        sampled = pd.concat([sampled,
                             remaining.sample(n=min(needed, len(remaining)), random_state=42)])
    return sampled.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate candidate entity pairs for manual annotation."
    )
    parser.add_argument("--entities",              required=True)
    parser.add_argument("--relations",             required=True)
    parser.add_argument("--output",                required=True)
    parser.add_argument("--top_k",                 type=int,   default=10)
    parser.add_argument("--semantic_floor",        type=float, default=0.65,
                        help="Similarity floor for non-proper-noun types (default: 0.65)")
    parser.add_argument("--semantic_floor_proper", type=float, default=0.88,
                        help="Similarity floor for Person/Conference/Journal (default: 0.88)")
    parser.add_argument("--max_pairs",             type=int,   default=3,
                        help="Max candidate pairs per entity (default: 3)")
    parser.add_argument("--model",                 default="all-mpnet-base-v2")
    parser.add_argument("--gpu",                   type=int,   default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    embedding_model = SentenceTransformer(args.model, device=device)

    df_entities = load_and_preprocess(args.entities, args.relations, embedding_model, device)
    df_pairs    = generate_candidate_pairs(
        df_entities, embedding_model, device,
        args.top_k, args.semantic_floor, args.semantic_floor_proper, args.max_pairs
    )
    print(f"Candidate pairs before sampling: {len(df_pairs)}")

    df_sample = adaptive_sample(df_pairs)
    df_sample.to_csv(args.output, index=False)
    print(f"Saved {len(df_sample)} pairs to {args.output}")


if __name__ == "__main__":
    main()
