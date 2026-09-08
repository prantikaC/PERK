# -*- coding: utf-8 -*-
"""
Compare two models' raw extraction against the golden triples set using the
SAME matching methodology as the paper's official extraction evaluation
(Section eval:ie): Sentence-BERT (all-mpnet-base-v2) embeddings of each
entity mention (label | context), dimension-wise max-pooling when an entity
is mentioned more than once within an email, cosine similarity, and Hungarian
one-to-one alignment per (email, entity type) at threshold tau=0.80.

This is a second, independent implementation from compare_extraction_recall.py
(which uses strict exact/token-subset text matching instead) -- run BOTH and
compare, rather than trusting either alone. Reuses compare_extraction_recall's
ModelIndex (entity/relation loading, mailNum resolution across the three
source-field conventions found in these pipelines) and its significance-test
functions (paired bootstrap, approximate randomization on F1), since those
parts are matching-method-agnostic.

Usage:
    python compare_extraction_hungarian.py \
        --golden ../../data/extraction_gold/refined_golden_set_target.csv \
        --golden_anchor_entities ../../data/entity_resolution/openai_v2/openai_v2_entities_final.csv \
        --model_a_name GPT --model_a_entities .../openai_v2_entities_final.csv --model_a_relations .../openai_v2_relations_final.csv \
        --model_b_name Gemma3_4B --model_b_entities .../entities_final.csv --model_b_relations .../relations_final.csv \
        --threshold 0.80 \
        --n_resamples 10000
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

import compare_extraction_recall as cer


def build_email_gold_entities(golden_df, email_to_mailnum):
    """mailnum -> type -> label -> [evidence_sentence contexts]"""
    per_mail = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for _, row in golden_df.iterrows():
        mn = email_to_mailnum.get(row["email_id"])
        if mn is None:
            continue
        per_mail[mn][row["subject_type"]][row["subject_label"]].append(str(row["evidence_sentence"]))
        per_mail[mn][row["object_type"]][row["object_label"]].append(str(row["evidence_sentence"]))
    return per_mail


def build_email_gold_triples(golden_df, email_to_mailnum):
    """mailnum -> set of (predicate, subj_type, subj_label, obj_type, obj_label)"""
    per_mail = defaultdict(set)
    for _, row in golden_df.iterrows():
        mn = email_to_mailnum.get(row["email_id"])
        if mn is None:
            continue
        per_mail[mn].add((row["predicate"], row["subject_type"], row["subject_label"],
                          row["object_type"], row["object_label"]))
    return per_mail


def build_email_predicted(model, relations_path, golden_mailnums):
    """Returns (per_mail_entities, per_mail_triples):
      per_mail_entities: mailnum -> type -> label -> [context strings]
      per_mail_triples:  mailnum -> list of (predicate, subj_type, subj_label, obj_type, obj_label)
    restricted to GOLD_PREDICATES and to golden-annotated emails, same scope
    rule as compare_extraction_recall.ModelIndex.predicted_triples_in_scope."""
    rel = pd.read_csv(relations_path)
    per_mail_entities = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_mail_triples = defaultdict(list)
    for _, row in rel.iterrows():
        predicate = row["relation"]
        if predicate not in cer.GOLD_PREDICATES:
            continue
        source = str(row.get("source", "") or "")
        mailnums = {model._resolve_source_token(s.strip()) for s in source.split(";") if s.strip()}
        mailnums.discard(None)
        scoped = mailnums & golden_mailnums
        if not scoped:
            continue
        start_id = str(row["start_id"]).strip()
        end_id = str(row["end_id"]).strip()
        context = str(row.get("context", "") or "")
        s_info = model.id_to_type_label.get(start_id)
        o_info = model.id_to_type_label.get(end_id)
        if not s_info or not o_info:
            continue
        s_type, s_label = s_info
        o_type, o_label = o_info
        for mn in scoped:
            per_mail_entities[mn][s_type][s_label].append(context)
            per_mail_entities[mn][o_type][o_label].append(context)
            per_mail_triples[mn].append((predicate, s_type, s_label, o_type, o_label))
    return per_mail_entities, per_mail_triples


def embed_entities(sbert, type_label_contexts, cache):
    """type_label_contexts: type -> label -> [contexts].
    Returns type -> [(label, pooled_embedding)], one entry per distinct
    (type, label), each mention's own embedding (label | context) computed
    then combined via dimension-wise MAX pooling across mentions, matching
    the paper's own construction exactly."""
    texts, keys = [], []
    for etype, labels in type_label_contexts.items():
        for label, contexts in labels.items():
            for ctx in (contexts or [""]):
                cache_key = (label, ctx)
                if cache_key not in cache:
                    texts.append(f"{label} | {ctx}")
                    keys.append((etype, label, cache_key))

    if texts:
        embs = sbert.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        for (etype, label, cache_key), emb in zip(keys, embs):
            cache[cache_key] = emb

    grouped = defaultdict(list)
    for etype, labels in type_label_contexts.items():
        for label, contexts in labels.items():
            for ctx in (contexts or [""]):
                grouped[(etype, label)].append(cache[(label, ctx)])

    out = defaultdict(list)
    for (etype, label), emb_list in grouped.items():
        pooled = np.max(np.stack(emb_list), axis=0)
        out[etype].append((label, pooled))
    return out


def align_email(gold_embedded, pred_embedded, threshold):
    """Hungarian one-to-one alignment per entity type, restricted to pairs
    with cosine similarity >= threshold. Returns (pred_type, pred_label) ->
    matched gold label, for accepted alignments only."""
    alignment = {}
    for etype in set(gold_embedded) & set(pred_embedded):
        g_list, p_list = gold_embedded[etype], pred_embedded[etype]
        if not g_list or not p_list:
            continue
        g_labels = [l for l, _ in g_list]
        g_embs = np.stack([e for _, e in g_list])
        p_labels = [l for l, _ in p_list]
        p_embs = np.stack([e for _, e in p_list])
        sim = p_embs @ g_embs.T  # cosine similarity, since embeddings are L2-normalized
        row_ind, col_ind = linear_sum_assignment(-sim)
        for r, c in zip(row_ind, col_ind):
            if sim[r, c] >= threshold:
                alignment[(etype, p_labels[r])] = g_labels[c]
    return alignment


def score_email(gold_triples, pred_triples, alignment):
    """After entity alignment, a predicted triple is correct iff its aligned
    (subject, predicate, object, types) exactly equals a gold triple.
    Returns (tp_gold, fn, tp_pred, fp) for this email."""
    matched_gold = set()
    tp_pred = fp = 0
    for predicate, s_type, s_label, o_type, o_label in pred_triples:
        aligned_s = alignment.get((s_type, s_label))
        aligned_o = alignment.get((o_type, o_label))
        if aligned_s is None or aligned_o is None:
            fp += 1
            continue
        key = (predicate, s_type, aligned_s, o_type, aligned_o)
        if key in gold_triples:
            tp_pred += 1
            matched_gold.add(key)
        else:
            fp += 1
    tp_gold = len(matched_gold)
    fn = len(gold_triples) - tp_gold
    return tp_gold, fn, tp_pred, fp


def score_model(model, relations_path, golden_df, email_to_mailnum, golden_mailnums,
                gold_entities_by_mail, gold_triples_by_mail, sbert, threshold, embed_cache):
    pred_entities_by_mail, pred_triples_by_mail = build_email_predicted(
        model, relations_path, golden_mailnums
    )
    stats = {}
    for mn in sorted(golden_mailnums):
        gold_embedded = embed_entities(sbert, gold_entities_by_mail.get(mn, {}), embed_cache)
        pred_embedded = embed_entities(sbert, pred_entities_by_mail.get(mn, {}), embed_cache)
        alignment = align_email(gold_embedded, pred_embedded, threshold)
        stats[mn] = score_email(
            gold_triples_by_mail.get(mn, set()),
            pred_triples_by_mail.get(mn, []),
            alignment,
        )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compare two extraction models via Sentence-BERT + Hungarian alignment "
                    "(same methodology as the paper's official IE evaluation), with "
                    "significance testing on the resulting F1."
    )
    parser.add_argument("--golden", required=True)
    parser.add_argument("--golden_anchor_entities", required=True)
    parser.add_argument("--model_a_name", required=True)
    parser.add_argument("--model_a_entities", required=True)
    parser.add_argument("--model_a_relations", required=True)
    parser.add_argument("--model_b_name", required=True)
    parser.add_argument("--model_b_entities", required=True)
    parser.add_argument("--model_b_relations", required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--embed_model", default="all-mpnet-base-v2")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Optional CSV of per-email (tp_gold,fn,tp_pred,fp) for both models")
    args = parser.parse_args()

    golden = pd.read_csv(args.golden)
    email_to_mailnum = cer.load_golden_mailnums(golden, args.golden_anchor_entities)
    golden_mailnums = {mn for mn in email_to_mailnum.values() if mn}

    gold_entities_by_mail = build_email_gold_entities(golden, email_to_mailnum)
    gold_triples_by_mail = build_email_gold_triples(golden, email_to_mailnum)

    model_a = cer.ModelIndex(args.model_a_name, args.model_a_entities, args.model_a_relations)
    model_b = cer.ModelIndex(args.model_b_name, args.model_b_entities, args.model_b_relations)

    import torch
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    sbert = SentenceTransformer(args.embed_model, device=device)
    embed_cache = {}

    stats_a_by_mail = score_model(model_a, args.model_a_relations, golden, email_to_mailnum,
                                   golden_mailnums, gold_entities_by_mail, gold_triples_by_mail,
                                   sbert, args.threshold, embed_cache)
    stats_b_by_mail = score_model(model_b, args.model_b_relations, golden, email_to_mailnum,
                                   golden_mailnums, gold_entities_by_mail, gold_triples_by_mail,
                                   sbert, args.threshold, embed_cache)

    mail_list = sorted(golden_mailnums)
    stats_a = np.array([stats_a_by_mail[mn] for mn in mail_list], dtype=float)
    stats_b = np.array([stats_b_by_mail[mn] for mn in mail_list], dtype=float)

    def summarize(stats):
        tp_gold, fn, tp_pred, fp = stats.sum(axis=0)
        recall = tp_gold / (tp_gold + fn) if (tp_gold + fn) else 0.0
        precision = tp_pred / (tp_pred + fp) if (tp_pred + fp) else 0.0
        f1 = cer._f1_from_sums(tp_gold, fn, tp_pred, fp)
        return precision, recall, f1, tp_pred, fp, tp_gold, fn

    prec_a, rec_a, f1_a, tp_pred_a, fp_a, tp_gold_a, fn_a = summarize(stats_a)
    prec_b, rec_b, f1_b, tp_pred_b, fp_b, tp_gold_b, fn_b = summarize(stats_b)

    print(f"Hungarian-alignment evaluation (threshold={args.threshold}, model={args.embed_model})")
    print(f"Scope: {len(golden_mailnums)} golden emails, predicates {sorted(cer.GOLD_PREDICATES)}\n")
    print(f"{args.model_a_name}: precision {prec_a:.4f} ({int(tp_pred_a)}/{int(tp_pred_a + fp_a)}), "
          f"recall {rec_a:.4f} ({int(tp_gold_a)}/{int(tp_gold_a + fn_a)}), F1 {f1_a:.4f}")
    print(f"{args.model_b_name}: precision {prec_b:.4f} ({int(tp_pred_b)}/{int(tp_pred_b + fp_b)}), "
          f"recall {rec_b:.4f} ({int(tp_gold_b)}/{int(tp_gold_b + fn_b)}), F1 {f1_b:.4f}")

    print("\n--- Paired Bootstrap Resampling on F1 (email-level) ---")
    mean_diff, (ci_low, ci_high), boot_p = cer.paired_bootstrap_f1(
        stats_a, stats_b, n_resamples=args.n_resamples, seed=args.seed
    )
    print(f"Mean resampled F1 diff: {mean_diff:.4f}  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"p-value (P[{args.model_b_name} >= {args.model_a_name}] under resampling): {boot_p:.4f}")

    print("\n--- Approximate Randomization on F1 (email-level) ---")
    observed_diff, perm_p = cer.approximate_randomization_f1(
        stats_a, stats_b, n_permutations=args.n_resamples, seed=args.seed
    )
    print(f"Observed F1 diff: {observed_diff:.4f}")
    print(f"Two-sided p-value: {perm_p:.4f}")

    if args.output:
        out_df = pd.DataFrame({
            "mailnum": mail_list,
            f"{args.model_a_name}_tp_gold": stats_a[:, 0], f"{args.model_a_name}_fn": stats_a[:, 1],
            f"{args.model_a_name}_tp_pred": stats_a[:, 2], f"{args.model_a_name}_fp": stats_a[:, 3],
            f"{args.model_b_name}_tp_gold": stats_b[:, 0], f"{args.model_b_name}_fn": stats_b[:, 1],
            f"{args.model_b_name}_tp_pred": stats_b[:, 2], f"{args.model_b_name}_fp": stats_b[:, 3],
        })
        out_df.to_csv(args.output, index=False)
        print(f"\nSaved per-email stats to {args.output}")


if __name__ == "__main__":
    main()
