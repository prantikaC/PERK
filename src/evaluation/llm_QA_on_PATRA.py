# -*- coding: utf-8 -*-
"""
Long-context QA baseline for PERK (no knowledge graph).

Feeds the entire PATRA email corpus to a long-context LLM (GPT-4.1) as a stable
system prefix and asks each PRASHNA-PATRA question directly; each answer is then
scored by an LLM judge (GPT-5.1) using the SAME `is_useful` rubric and
`not-available` short-circuit as `src/evaluation/kg_eval.py`. Because the judge is
identical, the accuracy here is directly comparable to the KG-QA numbers, and the
contrast isolates the value of the knowledge graph vs. brute-force long-context QA.

The corpus sits in a stable system prefix, so OpenAI's automatic prompt caching
reuses it across questions (per-question cost after the first is dominated by the
cached-input rate plus the short question/answer). The loop is checkpointed: results
are written after every question and re-running skips questions already answered.

Usage:
    python src/evaluation/llm_QA_on_PATRA.py \
        --corpus datasets/PATRA/PATRA.txt \
        --qa     datasets/PRASHNA_PATRA/PRASHNA_PATRA.csv \
        --output results/qa/longcontext_qa_results.csv \
        --fig    results/figures/qa/longcontext_qa_baseline.png
"""
import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

SYSTEM_PREFIX = (
    "You are a precise question-answering assistant. You are given a corpus of academic "
    "email threads (the PATRA dataset); individual emails are separated by the marker "
    "'EMAIL_END'. Answer the user's question using ONLY information found in these emails. "
    "Reply with the terse value(s) only (e.g. a name, date, or comma-separated list), or "
    "TRUE/FALSE for yes-no questions. If the answer is not present in the emails, reply "
    "exactly 'Not available'.\n\n"
    "=== PATRA CORPUS ===\n"
)


# --- LLM judge: identical rubric to kg_eval.py ------------------------------ #
class EvaluationResult(BaseModel):
    is_useful: bool = Field(
        description="True if the system answer provides the same correct information as the Gold Answer.")
    reason: str = Field(description="Brief explanation of the evaluation.")


def evaluate_answer(eval_llm, question: str, system_ans: str, gold_ans: str) -> EvaluationResult:
    gold_empty = "not available" in str(gold_ans).lower() or str(gold_ans).strip() == ""
    ans_empty = (system_ans.strip() == ""
                 or "not available" in system_ans.lower()
                 or "could not be extracted" in system_ans.lower())
    if gold_empty and ans_empty:
        return EvaluationResult(is_useful=True, reason="Both correctly identified missing information.")

    prompt = (
        "You are evaluating the usability of a Question Answering system.\n"
        f"User Question: {question}\n"
        f"Expected Gold Answer: {gold_ans}\n"
        f"System Answer: {system_ans}\n"
        "Does the system answer contain the correct information to satisfy the user's "
        "question compared to the Gold Answer?"
    )
    return eval_llm.with_structured_output(EvaluationResult).invoke(prompt)


def parse_args():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=str(repo / "datasets/PATRA/PATRA.txt"),
                   help="PATRA corpus given to the long-context model")
    p.add_argument("--qa", default=str(repo / "datasets/PRASHNA_PATRA/PRASHNA_PATRA.csv"),
                   help="PRASHNA-PATRA CSV with NLQ, GOLD_ANS, QUESTION_TYPE")
    p.add_argument("--gen_model", default="gpt-4.1", help="long-context answer generator")
    p.add_argument("--judge_model", default="gpt-5.1", help="LLM-as-judge (matches kg_eval.py)")
    p.add_argument("--sample_n", type=int, default=None,
                   help="evaluate only the first N questions (default: all)")
    p.add_argument("--output", default=str(repo / "results/qa/longcontext_qa_results.csv"))
    p.add_argument("--fig", default=str(repo / "results/figures/qa/longcontext_qa_baseline.png"))
    return p.parse_args()


def main():
    args = parse_args()
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set (put it in .env)"

    corpus = Path(args.corpus).read_text(encoding="utf-8")
    df = pd.read_csv(args.qa)[["NLQ", "GOLD_ANS", "QUESTION_TYPE"]].dropna(subset=["NLQ"]).reset_index(drop=True)
    if args.sample_n:
        df = df.head(args.sample_n).copy()
    print(f"Corpus: {len(corpus):,} chars (~{len(corpus)//4:,} tokens) | Questions: {len(df)}")
    print("By type:", df.QUESTION_TYPE.value_counts().to_dict())

    gen_llm = ChatOpenAI(model=args.gen_model, temperature=0)
    eval_llm = ChatOpenAI(model=args.judge_model, temperature=0)
    system_msg = SystemMessage(content=SYSTEM_PREFIX + corpus)   # stable prefix -> prompt caching

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if os.path.exists(args.output):
        done = {r["NLQ"]: r.to_dict() for _, r in pd.read_csv(args.output).iterrows()}
        print(f"Resuming: {len(done)} questions already done.")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        q, gold, qtype = row["NLQ"], str(row["GOLD_ANS"]), row["QUESTION_TYPE"]
        if q in done:
            rows.append(done[q]); continue
        try:
            ans = gen_llm.invoke([system_msg, HumanMessage(content=q)]).content.strip()
        except Exception as e:
            ans = f"[GEN ERROR] {e}"
        try:
            v = evaluate_answer(eval_llm, q, ans, gold)
            is_useful, reason = v.is_useful, v.reason
        except Exception as e:
            is_useful, reason = False, f"[JUDGE ERROR] {e}"
        rows.append({"NLQ": q, "GOLD_ANS": gold, "QUESTION_TYPE": qtype,
                     "GENERATED_ANS": ans, "IS_USEFUL": is_useful, "EVAL_REASON": reason})
        pd.DataFrame(rows).to_csv(args.output, index=False)     # checkpoint
        time.sleep(1)

    results = pd.read_csv(args.output)
    results["IS_USEFUL"] = results["IS_USEFUL"].astype(bool)
    overall = results["IS_USEFUL"].mean() * 100
    print(f"\nOverall accuracy: {overall:.2f}%  ({int(results.IS_USEFUL.sum())}/{len(results)})")
    by_type = results.groupby("QUESTION_TYPE")["IS_USEFUL"].agg(["mean", "count"])
    by_type["accuracy_%"] = (by_type["mean"] * 100).round(2)
    print(by_type[["accuracy_%", "count"]])

    acc = results.groupby("QUESTION_TYPE")["IS_USEFUL"].mean() * 100
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    ax = acc.plot.bar(rot=0, figsize=(7, 4), color="#5eead4", edgecolor="#334155")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title(f"Long-context {args.gen_model} QA baseline (judge: {args.judge_model})")
    for i, v in enumerate(acc):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center")
    plt.tight_layout()
    plt.savefig(args.fig)
    plt.savefig(str(Path(args.fig).with_suffix(".pdf")))
    print("Figure saved to", args.fig)


if __name__ == "__main__":
    main()
