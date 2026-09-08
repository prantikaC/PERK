# -*- coding: utf-8 -*-
"""
Generate candidate entity pairs for manual annotation, scoped to the entities
appearing in the golden IE triples file (refined_golden_set_target.csv) rather
than the full-corpus entities_final.csv.

Same FAISS semantic-blocking + lexical-gate approach as
generate_golden_candidates.py, adapted for a flat triples input that has no
entity IDs and no JSON properties: each distinct (label, type) pair across
subject/object columns becomes one entity, with context built by
concatenating the evidence_sentence of every triple it appears in. There is
no strict-metadata check here (no taskDate/personEmail fields in this file).
Output is a CSV for annotation (label column left blank), same schema as
generate_golden_candidates.py's output.

Usage:
    python generate_golden_candidates_from_triples.py \
        --triples refined_golden_set_target.csv \
        --output golden_triples_entity_pair_for_annotation.csv \
        --top_k 10 \
        --semantic_floor 0.65 \
        --semantic_floor_proper 0.88 \
        --max_pairs 3
"""

import argparse
from collections import defaultdict

import faiss
import jellyfish
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PROPER_NOUN_TYPES = ['Person', 'Conference', 'Journal']


def load_and_preprocess(triples_path):
    df = pd.read_csv(triples_path)

    ctx_map = defaultdict(list)
    for _, row in df.iterrows():
        ctx = str(row['evidence_sentence']).strip()
        if not ctx or ctx.lower() == "nan":
            continue
        subj_key = (str(row['subject_label']).strip(), str(row['subject_type']).strip())
        obj_key = (str(row['object_label']).strip(), str(row['object_type']).strip())
        ctx_map[subj_key].append(ctx)
        ctx_map[obj_key].append(ctx)

    rows = []
    for (label, etype), ctxs in ctx_map.items():
        if not label or label.lower() == "nan" or not etype or etype.lower() == "nan":
            continue
        # Dedupe repeated identical evidence sentences, cap to keep embeddings sane.
        seen, deduped = set(), []
        for c in ctxs:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        rows.append({"label": label, "type": etype, "context": " | ".join(deduped[:5])})

    df_entities = pd.DataFrame(rows)
    df_entities['id'] = [f"{row.type}{i}" for i, row in enumerate(df_entities.itertuples(), start=1)]
    df_entities['text_to_embed'] = df_entities.apply(
        lambda x: f"Entity: {x['label']}. Evidence: {x['context']}", axis=1
    )
    print(f"Built {len(df_entities)} distinct entities from golden triples.")
    return df_entities


def generate_candidate_pairs(df_entities, model, top_k, semantic_floor,
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
        description="Generate candidate entity pairs from the golden IE triples file."
    )
    parser.add_argument("--triples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--semantic_floor", type=float, default=0.65)
    parser.add_argument("--semantic_floor_proper", type=float, default=0.88)
    parser.add_argument("--max_pairs", type=int, default=3)
    parser.add_argument("--model", default="all-mpnet-base-v2")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    embedding_model = SentenceTransformer(args.model, device=device)

    df_entities = load_and_preprocess(args.triples)
    df_pairs = generate_candidate_pairs(
        df_entities, embedding_model,
        args.top_k, args.semantic_floor, args.semantic_floor_proper, args.max_pairs
    )
    print(f"Candidate pairs before sampling: {len(df_pairs)}")
    df_sample = adaptive_sample(df_pairs)
    df_sample.to_csv(args.output, index=False)
    print(f"Saved {len(df_sample)} pairs to {args.output}")


if __name__ == "__main__":
    main()
