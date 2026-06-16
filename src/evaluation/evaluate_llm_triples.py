# -*- coding: utf-8 -*-
"""
Sentence-BERT triple evaluation (script version of Evaluate_LLM_Triples.ipynb).

SBERT (all-mpnet-base-v2) entity embeddings with context, max-pooled, Hungarian
alignment at threshold tau, then Subject / Object / All-Entities / Predicate /
Triple micro P/R/F1 for both context modes (evidence_sentence, full_email_context).
Triples are filtered by ontology domain/range validity.

Input is a model's comparison_triples.csv (build it with build_comparison_triples.py).

Usage:
    python src/evaluation/evaluate_llm_triples.py \
        --name GptOss \
        --pred results/extractions/gptoss/evaluation_triples/comparison_triples.csv \
        --tau  0.80 --gpu 0
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# --- Pin the GPU BEFORE importing torch ------------------------------------ #
# torch reads CUDA_VISIBLE_DEVICES exactly once, at import time; setting it
# afterwards has no effect. We therefore parse --gpu from argv here, mask all
# other GPUs, and force PCI-bus ordering so --gpu N is PHYSICAL GPU N (the same
# number nvidia-smi shows). After masking, the chosen card is the only visible
# device, so inside the program it is always cuda:0 -- no other GPU can be touched.
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

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

MODEL_NAME = "all-mpnet-base-v2"
THRESHOLDS = np.arange(0.60, 0.96, 0.05)

VALID_RELATIONS = {
    "hasAuthor": ({"Paper"}, {"Person"}),
    "identifies": ({"SubmissionID"}, {"Paper"}),
    "inVenue": ({"SubmissionID"}, {"Conference", "Journal"}),
    "movesTo": ({"PaperStatus", "SubmissionID"}, {"PaperStatus"}),
    "hasPaperInfo": ({"Paper"}, {"PaperBib"}),
    "worksWith": ({"Person"}, {"Method", "Dataset", "Metric"}),
    "worksOn": ({"Person"}, {"Task"}),
    "usedFor": ({"Dataset", "Method"}, {"Task"}),
    "evaluates": ({"Metric"}, {"Task", "Method", "Dataset"}),
    "uses": ({"Method"}, {"Dataset"}),
    "attends": ({"Person"}, {"Meeting", "Conference"}),
}


def calc_f1(tp, pred_total, gold_total):
    p = tp / pred_total if pred_total > 0 else 0
    r = tp / gold_total if gold_total > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


def domain_range_valid(s_type, predicate, o_type):
    if predicate not in VALID_RELATIONS:
        return False
    domain, range_ = VALID_RELATIONS[predicate]
    return s_type in domain and o_type in range_


def safe_string(value):
    if pd.isna(value):
        return None
    return str(value).strip()


def build_entity_dict(df, context_mode, email_context_map):
    entity_data = {}
    for _, row in df.iterrows():
        if pd.isna(row["email_id"]):
            continue
        eid = str(row["email_id"])
        entity_data.setdefault(eid, {})
        if context_mode == "full_email_context":
            context = email_context_map.get(row["email_id"], "")
        else:
            context = "" if pd.isna(row["evidence_sentence"]) else str(row["evidence_sentence"])
        for label_col, type_col in [("subject_label", "subject_type"),
                                    ("object_label", "object_type")]:
            label = safe_string(row[label_col])
            etype = safe_string(row[type_col])
            if label is None or etype is None:
                continue
            key = (label.lower(), etype)
            entity_data[eid].setdefault(key, {"label": label, "type": etype, "contexts": []})
            entity_data[eid][key]["contexts"].append(context)
    return entity_data


def max_pool_embeddings(entity_dict, model, batch_size):
    pooled = {}
    for key, data in entity_dict.items():
        texts = [f"Entity: {data['label']} | Context: {c}" for c in data["contexts"]]
        embs = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        pooled[key] = np.max(embs, axis=0)
    return pooled


def precompute_similarity(g_rows, p_rows, context_mode, email_context_map, model, batch_size):
    eid = str(g_rows.iloc[0]["email_id"])
    g_entities = build_entity_dict(g_rows, context_mode, email_context_map).get(eid, {})
    p_entities = build_entity_dict(p_rows, context_mode, email_context_map).get(eid, {})
    if not g_entities or not p_entities:
        return None
    g_emb = max_pool_embeddings(g_entities, model, batch_size)
    p_emb = max_pool_embeddings(p_entities, model, batch_size)
    g_keys, p_keys = list(g_emb.keys()), list(p_emb.keys())
    g_matrix = np.stack([g_emb[k] for k in g_keys])
    p_matrix = np.stack([p_emb[k] for k in p_keys])
    sim = cosine_similarity(g_matrix, p_matrix)
    return {"gold_keys": g_keys, "pred_keys": p_keys, "similarity": sim}


def align_from_precomputed(precomputed, tau):
    sim = precomputed["similarity"]
    g_keys, p_keys = precomputed["gold_keys"], precomputed["pred_keys"]
    row_ind, col_ind = linear_sum_assignment(-sim)
    alignment = {}
    for r, c in zip(row_ind, col_ind):
        if sim[r, c] >= tau:
            alignment[p_keys[c]] = g_keys[r]
    return alignment


def evaluate_model(name, pred_file, golden_file, out_dir, model, batch_size, taus):
    gold_df = pd.read_csv(golden_file)
    pred_df = pd.read_csv(pred_file)
    email_context_map = (gold_df.dropna(subset=["full_email_context"])
                         .drop_duplicates("email_id")
                         .set_index("email_id")["full_email_context"].to_dict())

    results = {}
    for context_mode in ["evidence_sentence", "full_email_context"]:
        print(f"\nEvaluating {name} | Context={context_mode}")
        precomputed_all = {}
        for eid in set(gold_df.email_id).intersection(pred_df.email_id):
            g_rows = gold_df[gold_df.email_id == eid]
            p_rows = pred_df[pred_df.email_id == eid]
            res = precompute_similarity(g_rows, p_rows, context_mode,
                                        email_context_map, model, batch_size)
            if res:
                precomputed_all[eid] = res

        log_path = os.path.join(out_dir, f"{name}_{context_mode}_metrics.log")
        rows_out = {}
        with open(log_path, "w") as log:
            log.write(f"Model: {name}\nContext: {context_mode}\nTimestamp: {datetime.now()}\n")
            for tau in taus:
                subj_tp = subj_pt = subj_gt = 0
                obj_tp = obj_pt = obj_gt = 0
                ent_tp = ent_pt = ent_gt = 0
                pr_tp = pr_pt = pr_gt = 0
                tr_tp = tr_pt = tr_gt = 0
                for eid, pc in precomputed_all.items():
                    g_rows = gold_df[gold_df.email_id == eid]
                    p_rows = pred_df[pred_df.email_id == eid]
                    alignment = align_from_precomputed(pc, tau)
                    matched_gold = set(alignment.values())

                    g_subj, g_obj, p_subj, p_obj = set(), set(), set(), set()
                    for _, r in g_rows.iterrows():
                        sl, ol = safe_string(r.subject_label), safe_string(r.object_label)
                        st, ot = safe_string(r.subject_type), safe_string(r.object_type)
                        if None not in [sl, st]: g_subj.add((sl.lower(), st))
                        if None not in [ol, ot]: g_obj.add((ol.lower(), ot))
                    for _, r in p_rows.iterrows():
                        sl, ol = safe_string(r.subject_label), safe_string(r.object_label)
                        st, ot = safe_string(r.subject_type), safe_string(r.object_type)
                        if None not in [sl, st]: p_subj.add((sl.lower(), st))
                        if None not in [ol, ot]: p_obj.add((ol.lower(), ot))

                    subj_tp += len(matched_gold & g_subj); subj_pt += len(p_subj); subj_gt += len(g_subj)
                    obj_tp += len(matched_gold & g_obj); obj_pt += len(p_obj); obj_gt += len(g_obj)
                    g_ent, p_ent = g_subj | g_obj, p_subj | p_obj
                    ent_tp += len(matched_gold & g_ent); ent_pt += len(p_ent); ent_gt += len(g_ent)

                    g_preds, p_preds = set(g_rows.predicate.dropna()), set(p_rows.predicate.dropna())
                    pr_tp += len(g_preds & p_preds); pr_pt += len(p_preds); pr_gt += len(g_preds)

                    aligned_pred_triples = set()
                    for _, r in p_rows.iterrows():
                        sl, ol = safe_string(r.subject_label), safe_string(r.object_label)
                        st, ot, pd_ = safe_string(r.subject_type), safe_string(r.object_type), safe_string(r.predicate)
                        if None in [sl, ol, st, ot, pd_] or not domain_range_valid(st, pd_, ot):
                            continue
                        s = alignment.get((sl.lower(), st), (sl.lower(), st))
                        o = alignment.get((ol.lower(), ot), (ol.lower(), ot))
                        aligned_pred_triples.add((s, pd_, o))
                    gold_triples = set()
                    for _, r in g_rows.iterrows():
                        sl, ol = safe_string(r.subject_label), safe_string(r.object_label)
                        st, ot, pd_ = safe_string(r.subject_type), safe_string(r.object_type), safe_string(r.predicate)
                        if None in [sl, ol, st, ot, pd_] or not domain_range_valid(st, pd_, ot):
                            continue
                        gold_triples.add(((sl.lower(), st), pd_, (ol.lower(), ot)))
                    tr_tp += len(aligned_pred_triples & gold_triples)
                    tr_pt += len(aligned_pred_triples); tr_gt += len(gold_triples)

                m = {
                    "Subject": calc_f1(subj_tp, subj_pt, subj_gt),
                    "Object": calc_f1(obj_tp, obj_pt, obj_gt),
                    "All Entities": calc_f1(ent_tp, ent_pt, ent_gt),
                    "Predicate": calc_f1(pr_tp, pr_pt, pr_gt),
                    "Triple": calc_f1(tr_tp, tr_pt, tr_gt),
                }
                rows_out[round(float(tau), 2)] = m
                log.write(f"\nTau={tau:.2f}\n")
                for k, (p, r, f) in m.items():
                    log.write(f"{k:12s} P={p:.4f} R={r:.4f} F1={f:.4f}\n")
        results[context_mode] = rows_out
        print(f"  log -> {log_path}")
    return results


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--golden", default=str(repo / "datasets/extraction_gold/refined_golden_set_target.csv"))
    ap.add_argument("--out_dir", default=str(repo / "results/evaluation"))
    ap.add_argument("--tau", type=float, default=0.80)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    # Only the pinned card is visible, so it is cuda:0 inside this process.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Pinned to PHYSICAL GPU {_PINNED_GPU} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}) | model: {MODEL_NAME}")
    if device != "cpu":
        print(f"  -> {torch.cuda.get_device_name(0)}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    res = evaluate_model(args.name, args.pred, args.golden, args.out_dir,
                         model, args.batch_size, THRESHOLDS)

    tau = round(args.tau, 2)
    print(f"\n================ {args.name}  @ tau={tau:.2f} ================")
    print(f"{'metric':14s} {'context=evidence':>24s}   {'context=full_email':>24s}")
    print(f"{'':14s} {'P':>7s}{'R':>8s}{'F1':>8s}   {'P':>8s}{'R':>8s}{'F1':>8s}")
    for metric in ["Subject", "Object", "All Entities", "Predicate", "Triple"]:
        ev = res["evidence_sentence"][tau][metric]
        fe = res["full_email_context"][tau][metric]
        print(f"{metric:14s} {ev[0]:7.3f}{ev[1]:8.3f}{ev[2]:8.3f}   "
              f"{fe[0]:8.3f}{fe[1]:8.3f}{fe[2]:8.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
