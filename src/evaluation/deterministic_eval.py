# -*- coding: utf-8 -*-
"""
Deterministic (no-LLM) re-judging of KG-QA eval results.

Takes a results CSV with GOLD_ANS and DB_OUTPUT columns -- either
KG_UTILITY_RESULTS_*.csv from kg_eval_v4.py (has an LLM-judge column too,
for comparison) or KG_QA_ANSWERS_*.csv from kg_qa.py (answers only, no
judge) -- and re-scores each row with pure string/value matching against
the gold answer. No API calls, fully reproducible, zero non-determinism.

For boolean and "Not available"/empty gold answers there's nothing to take
a partial subset of, so all three metrics below collapse to one shared
verdict. For list-shaped gold answers, gold and the DB output are each
split into a set of items (fuzzy substring match, case/hyphen-insensitive)
and compared in THREE distinct ways, each written to its own column:

  - EXACT_MATCH:            gold's items and the DB output's items cover
                             exactly the same things (both directions hold).
  - ANSWER_CONTAINS_GOLD:   every gold item is found in the DB output --
                             recall on gold; the DB output may say MORE than
                             gold and still pass (extra correct info is fine).
                             This is the metric closest to how the LLM judge
                             itself scores usefulness.
  - ANSWER_SUBSET_OF_GOLD:  every DB-output value is found in gold --
                             precision; the DB output introduces nothing
                             unsupported, but may still be missing some gold
                             items and still pass.

This is intentionally stricter/dumber than an LLM judge: it cannot tell that
two paraphrases mean the same thing, so it will under-count some correct
answers (false negatives) but never over-forgive garbled output. Best used
ALONGSIDE the LLM-judged IS_USEFUL column (preserved when present) to see
where the two disagree and pull rows worth a manual look.

Outputs Pass/Fail (matching the LLM judge's own vocabulary) rather than 1/0,
plus a log file in the same per-question block style as kg_eval_v4.py's log,
showing all three verdicts side by side and calling out every disagreement
with the LLM judge (compared against ANSWER_CONTAINS_GOLD) with both sides'
reasoning.

Usage:
    python deterministic_eval.py --input eval_runs/KG_QA_ANSWERS_....csv
    python deterministic_eval.py --input eval_runs/KG_UTILITY_RESULTS_....csv
    python deterministic_eval.py --input ... --output out.csv --log out.txt
"""

import argparse
import ast
import re

import pandas as pd


def _split_top_level(text, sep=","):
    """Split on `sep` but only outside parentheses/brackets, so parenthetical
    clarifications in a gold answer (e.g. "X (details, more details)") don't
    get split into spurious extra items."""
    parts, buf, depth = [], [], 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def parse_db_output(db_output_str):
    """Best-effort parse of the stringified Neo4j result list into Python
    objects. Returns None if it isn't a parseable list-of-dicts (e.g. it's
    one of the free-text sentinel strings like "The information could not
    be extracted...")."""
    s = str(db_output_str).strip()
    if s.endswith("...") :  # truncated by kg_eval_v4's log/CSV writer
        s = s[:-3]
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def flatten_values(parsed_rows):
    """Every leaf value across every row/dict, as strings."""
    out = []
    for row in parsed_rows:
        if isinstance(row, dict):
            for v in row.values():
                if v is not None:
                    out.append(str(v))
        else:
            out.append(str(row))
    return out


NEGATIVE_TEXT_MARKERS = (
    "not available", "could not be extracted", "unanswerable",
    "error during execution", "no valid schema path",
)


def is_db_empty(db_output_str, parsed_rows):
    s = str(db_output_str).strip().lower()
    if s in ("[]", "", "none"):
        return True
    if any(m in s for m in NEGATIVE_TEXT_MARKERS):
        return True
    if parsed_rows is None:
        return False
    if not parsed_rows:
        return True
    for row in parsed_rows:
        if not isinstance(row, dict) or not row:
            return False
        if any(v not in (None, False) for v in row.values()):
            return False
    return True


GOLD_NEGATIVE_MARKERS = ("not available", "not answerable", "unanswerable", "no answer")


def is_gold_empty(gold):
    g = str(gold).strip().lower()
    return g == "" or any(m in g for m in GOLD_NEGATIVE_MARKERS)


BOOL_TRUE = {"true", "yes"}
BOOL_FALSE = {"false", "no"}


def gold_bool(gold):
    g = str(gold).strip().lower()
    if g in BOOL_TRUE:
        return True
    if g in BOOL_FALSE:
        return False
    return None


def db_bool(parsed_rows, db_output_str):
    """Look for an explicit boolean result in the parsed rows; fall back to
    scanning the raw string for a lone True/False token."""
    if parsed_rows:
        for row in parsed_rows:
            if isinstance(row, dict):
                for v in row.values():
                    if isinstance(v, bool):
                        return v
            elif isinstance(row, bool):
                return row
    s = str(db_output_str)
    if re.search(r"\bTrue\b", s):
        return True
    if re.search(r"\bFalse\b", s):
        return False
    return None


def normalize(text):
    """Lowercase, collapse whitespace, and treat hyphens/underscores as
    spaces so e.g. "F1-score" and "F1 score" compare equal -- this is a
    common source of false negatives since the gold answers and DB output
    are written by different processes with different punctuation habits."""
    s = str(text).strip().lower()
    s = re.sub(r"[-_]", " ", s)
    return re.sub(r"\s+", " ", s)


def _items_covered(needed, available):
    """True iff every item in `needed` has a fuzzy (substring, either
    direction) match somewhere in `available`. Returns (bool, missing_list)."""
    missing = [n for n in needed if not any(n in a or a in n for a in available if a)]
    return (len(missing) == 0), missing


def match_list_three_way(gold, parsed_rows):
    """Compare gold items against DB-output values in both directions:
      - CONTAINS_GOLD: every gold item is found in the DB output (recall --
        the DB output may say MORE than gold, that's fine, tolerant of extra
        correct content).
      - SUBSET_OF_GOLD: every DB-output value is found in gold (precision --
        the DB output says NOTHING beyond what gold lists; it may still be
        missing some gold items though).
      - EXACT: both directions hold (the DB output and gold cover exactly
        the same set of items, modulo fuzzy substring matching).
    Returns a dict: {'exact': (bool, reason), 'contains_gold': (bool, reason),
    'subset_of_gold': (bool, reason)}."""
    gold_items = [normalize(x) for x in _split_top_level(str(gold)) if normalize(x)]
    db_values = [normalize(v) for v in flatten_values(parsed_rows or [])]

    if not gold_items:
        return {
            "exact": (True, "Empty gold item list."),
            "contains_gold": (True, "Empty gold item list."),
            "subset_of_gold": (True, "Empty gold item list."),
        }
    if not db_values:
        reason = "DB output has no values to match against a non-empty gold answer."
        return {
            "exact": (False, reason),
            "contains_gold": (False, reason),
            "subset_of_gold": (False, reason),
        }

    gold_covered, missing_from_db = _items_covered(gold_items, db_values)
    db_covered, extra_in_db = _items_covered(db_values, gold_items)

    contains_gold = (gold_covered,
        "All gold item(s) found in DB output (substring match)." if gold_covered
        else f"Gold item(s) not found in DB output: {missing_from_db}")
    subset_of_gold = (db_covered,
        "All DB-output value(s) are accounted for in gold (no unsupported extras)." if db_covered
        else f"DB-output value(s) not found in gold (unsupported/extra content): {extra_in_db}")
    exact = (gold_covered and db_covered,
        "Gold and DB output cover exactly the same items." if (gold_covered and db_covered)
        else "Not an exact match -- see contains_gold/subset_of_gold reasons.")

    return {"exact": exact, "contains_gold": contains_gold, "subset_of_gold": subset_of_gold}


def _list_confusion_counts(gold, parsed_rows):
    """(tp, fp, fn) for a list-shaped gold answer: tp = gold items found in the
    DB output, fn = gold items missing from it, fp = DB-output values not
    accounted for in gold (unsupported/extra content)."""
    gold_items = [normalize(x) for x in _split_top_level(str(gold)) if normalize(x)]
    db_values = [normalize(v) for v in flatten_values(parsed_rows or [])]

    if not gold_items:
        return (0, len(set(db_values)), 0)
    if not db_values:
        return (0, 0, len(gold_items))

    _, missing_from_db = _items_covered(gold_items, db_values)
    _, extra_in_db = _items_covered(db_values, gold_items)
    tp = len(gold_items) - len(missing_from_db)
    return (tp, len(extra_in_db), len(missing_from_db))


def row_confusion_counts(gold, db_output_str):
    """(tp, fp, fn) for ONE row, unified across boolean/empty/list gold shapes,
    used to compute precision/recall/F1 alongside the Pass/Fail verdicts.

    A row that is a correctly-predicted true negative (gold says False/Not
    available, DB output correctly says so too) contributes (0, 0, 0) -- it
    is neither a hit, a false alarm, nor a miss, so it doesn't affect
    precision/recall directly; it is instead given full credit (F1 = 1.0)
    when averaging per-row F1 scores (see macro_f1 in main()).
    """
    parsed = parse_db_output(db_output_str)

    gb = gold_bool(gold)
    if gb is not None:
        db_b = db_bool(parsed, db_output_str)
        if db_b is not None:
            if gb is True:
                return (1, 0, 0) if db_b is True else (0, 0, 1)
            else:
                return (0, 0, 0) if db_b is False else (0, 1, 0)
        empty_db = is_db_empty(db_output_str, parsed)
        if gb is True:
            return (1, 0, 0) if not empty_db else (0, 0, 1)  # not empty -> treated as TRUE -> hit
        else:
            return (0, 0, 0) if empty_db else (0, 1, 0)

    empty_db = is_db_empty(db_output_str, parsed)
    empty_gold = is_gold_empty(gold)
    if empty_gold and empty_db:
        return (0, 0, 0)
    if empty_gold and not empty_db:
        return (0, 1, 0)
    if not empty_gold and empty_db:
        return (0, 0, 1)

    return _list_confusion_counts(gold, parsed)


def three_way_verdict(gold, db_output_str):
    """Same special-case handling (boolean gold, empty/'Not available' gold)
    as the original single-verdict logic, but returns all three directional
    verdicts. For boolean and empty-gold cases there's no meaningful subset
    structure (a single yes/no or an empty answer has nothing to be a
    'partial' match of), so all three verdicts collapse to the same value --
    only list-shaped gold answers can actually differ across the three."""
    parsed = parse_db_output(db_output_str)

    gb = gold_bool(gold)
    if gb is not None:
        db_b = db_bool(parsed, db_output_str)
        if db_b is not None:
            v, r = (gb == db_b), f"Gold={gb}, DB={db_b} (explicit boolean)."
        else:
            empty_db = is_db_empty(db_output_str, parsed)
            if gb is True:
                v, r = (not empty_db), (
                    "No explicit boolean; DB has positive-evidence content (treated as TRUE)."
                    if not empty_db else
                    "Gold TRUE but DB output is empty (no evidence found)."
                )
            else:
                v, r = empty_db, (
                    "No explicit boolean; DB output is empty (treated as FALSE/no evidence)."
                    if empty_db else
                    "Gold FALSE but DB output has content (contradicts, or is unrelated evidence)."
                )
        return {"exact": (v, r), "contains_gold": (v, r), "subset_of_gold": (v, r)}

    empty_db = is_db_empty(db_output_str, parsed)
    empty_gold = is_gold_empty(gold)
    if empty_gold and empty_db:
        v, r = True, "Both gold and DB output are empty/not-available."
        return {"exact": (v, r), "contains_gold": (v, r), "subset_of_gold": (v, r)}
    if empty_gold and not empty_db:
        v, r = False, "Gold is 'not available' but DB output has content (fabrication)."
        return {"exact": (v, r), "contains_gold": (v, r), "subset_of_gold": (v, r)}
    if not empty_gold and empty_db:
        v, r = False, "Gold has content but DB output is empty/not-available."
        return {"exact": (v, r), "contains_gold": (v, r), "subset_of_gold": (v, r)}

    return match_list_three_way(gold, parsed)


def deterministic_verdict(gold, db_output_str):
    """Backward-compatible single-verdict entry point -- equivalent to the
    'contains_gold' (recall) direction, which was this script's original
    (and default) matching behavior."""
    return three_way_verdict(gold, db_output_str)["contains_gold"]


def _pf(is_pass):
    return "Pass" if is_pass else "Fail"


def _prf1(tp, fp, fn):
    """(precision, recall, f1) for one row's (tp, fp, fn). A true-negative
    row (0, 0, 0) -- nothing was expected and nothing extraneous was
    produced -- gets full credit (1.0, 1.0, 1.0) by convention, matching its
    Pass verdict elsewhere, rather than being left undefined (0/0)."""
    if tp == 0 and fp == 0 and fn == 0:
        return (1.0, 1.0, 1.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return (precision, recall, f1)


def main():
    p = argparse.ArgumentParser(description="Deterministic (no-LLM) re-judging of KG-QA eval results.")
    p.add_argument("--input", required=True, help="KG_UTILITY_RESULTS_*.csv from kg_eval_v4.py")
    p.add_argument("--output", default=None, help="Output CSV path (default: input with _det suffix)")
    p.add_argument("--log", default=None, help="Output log path (default: input's LOG counterpart with _det suffix)")
    args = p.parse_args()

    df = pd.read_csv(args.input)

    exact_v, exact_r = [], []
    contains_v, contains_r = [], []
    subset_v, subset_r = [], []
    tps, fps, fns = [], [], []
    precisions, recalls, f1s = [], [], []
    for _, row in df.iterrows():
        gold, db_out = row.get("GOLD_ANS", ""), row.get("DB_OUTPUT", "")
        res = three_way_verdict(gold, db_out)
        v, r = res["exact"]; exact_v.append(v); exact_r.append(r)
        v, r = res["contains_gold"]; contains_v.append(v); contains_r.append(r)
        v, r = res["subset_of_gold"]; subset_v.append(v); subset_r.append(r)

        tp, fp, fn = row_confusion_counts(gold, db_out)
        tps.append(tp); fps.append(fp); fns.append(fn)
        p, rc, f1 = _prf1(tp, fp, fn)
        precisions.append(p); recalls.append(rc); f1s.append(f1)

    df["EXACT_MATCH"] = [_pf(v) for v in exact_v]
    df["EXACT_MATCH_REASON"] = exact_r
    df["ANSWER_CONTAINS_GOLD"] = [_pf(v) for v in contains_v]
    df["ANSWER_CONTAINS_GOLD_REASON"] = contains_r
    df["ANSWER_SUBSET_OF_GOLD"] = [_pf(v) for v in subset_v]
    df["ANSWER_SUBSET_OF_GOLD_REASON"] = subset_r
    df["PRECISION"] = precisions
    df["RECALL"] = recalls
    df["F1"] = f1s

    # Kept for backward compatibility with anything reading the old columns --
    # equivalent to ANSWER_CONTAINS_GOLD (this script's original default).
    df["DETERMINISTIC_VERDICT"] = df["ANSWER_CONTAINS_GOLD"]
    df["DETERMINISTIC_REASON"] = df["ANSWER_CONTAINS_GOLD_REASON"]

    n = len(df)
    print(f"Exact-match accuracy:            {sum(exact_v)/n:.2%} ({sum(exact_v)}/{n})")
    print(f"Answer-contains-gold accuracy:    {sum(contains_v)/n:.2%} ({sum(contains_v)}/{n})  (recall: gold fully covered by answer, extras OK)")
    print(f"Answer-subset-of-gold accuracy:   {sum(subset_v)/n:.2%} ({sum(subset_v)}/{n})  (precision: answer has nothing beyond gold)")

    # Macro: each row's own P/R/F1 computed independently, then averaged with
    # equal weight per question regardless of how many items its answer has
    # (the SQuAD convention -- a boolean question counts the same as a
    # 5-item list question).
    macro_p, macro_r, macro_f1 = sum(precisions)/n, sum(recalls)/n, sum(f1s)/n
    # Micro: pool raw TP/FP/FN across every row first, then compute one
    # global P/R/F1 -- rows with more gold/DB items implicitly count more.
    sum_tp, sum_fp, sum_fn = sum(tps), sum(fps), sum(fns)
    if sum_tp == 0 and sum_fp == 0 and sum_fn == 0:
        micro_p, micro_r, micro_f1 = 1.0, 1.0, 1.0
    else:
        micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
        micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    print(f"\nMacro P/R/F1 (per-question average): P={macro_p:.2%}  R={macro_r:.2%}  F1={macro_f1:.2%}")
    print(f"Micro P/R/F1 (pooled item counts):    P={micro_p:.2%}  R={micro_r:.2%}  F1={micro_f1:.2%}  (TP={sum_tp}, FP={sum_fp}, FN={sum_fn})")

    has_llm = "IS_USEFUL" in df.columns
    if has_llm:
        llm_pass = df["IS_USEFUL"].astype(str).str.strip().str.lower() == "true"
        # LLM agreement is compared against ANSWER_CONTAINS_GOLD, since that's
        # the closest analog to how the LLM judge itself scores usefulness
        # (tolerant of extra correct content, only penalizes missing gold items).
        det_pass = df["ANSWER_CONTAINS_GOLD"] == "Pass"
        mismatch = llm_pass != det_pass
        agree = (~mismatch).sum()
        print(f"\nAgreement with LLM-judge column (vs. answer-contains-gold): {agree}/{n} ({agree/n:.2%})")
        print(f"\n{mismatch.sum()} disagreements (LLM vs deterministic):")
        for _, row in df[mismatch].head(20).iterrows():
            print(f"  Q: {str(row['NLQ'])[:70]}")
            print(f"     Gold: {str(row['GOLD_ANS'])[:70]}")
            llm_v = "Pass" if str(row["IS_USEFUL"]).strip().lower() == "true" else "Fail"
            print(f"     LLM-judge={llm_v}  Deterministic={row['ANSWER_CONTAINS_GOLD']} ({row['ANSWER_CONTAINS_GOLD_REASON']})")

    out_path = args.output or re.sub(r"\.csv$", "_det.csv", args.input)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # --- log file, same per-question block style as kg_eval_v4.py's log ---
    log_path = args.log
    if log_path is None:
        base = re.sub(r"\.csv$", "", args.input)
        base = re.sub(r"KG_UTILITY_RESULTS_", "KG_UTILITY_LOG_", base)
        base = re.sub(r"KG_QA_ANSWERS_", "KG_QA_LOG_", base)
        log_path = base + "_det.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Deterministic re-judging of {args.input} on {len(df)} questions...\n\n")
        for i, row in df.iterrows():
            f.write(f"--- {i+1}/{len(df)} | Q: {row['NLQ']} ---\n")
            f.write(f"Gold Answer: {row['GOLD_ANS']}\n")
            db_out = str(row.get("DB_OUTPUT", ""))
            f.write(f"DB Output: {db_out[:150]}{'...' if len(db_out) > 150 else ''}\n")

            if has_llm:
                llm_verdict = "Pass" if str(row["IS_USEFUL"]).strip().lower() == "true" else "Fail"
                llm_reason = row.get("EVAL_REASON", "")
                f.write(f"LLM Verdict: {llm_verdict} ({llm_reason})\n")

            f.write(f"Exact Match:          {row['EXACT_MATCH']} ({row['EXACT_MATCH_REASON']})\n")
            f.write(f"Answer Contains Gold: {row['ANSWER_CONTAINS_GOLD']} ({row['ANSWER_CONTAINS_GOLD_REASON']})\n")
            f.write(f"Answer Subset of Gold:{row['ANSWER_SUBSET_OF_GOLD']} ({row['ANSWER_SUBSET_OF_GOLD_REASON']})\n")

            if has_llm and llm_verdict != row["ANSWER_CONTAINS_GOLD"]:
                f.write(f"  >>> MISMATCH: LLM={llm_verdict}, Deterministic(answer-contains-gold)={row['ANSWER_CONTAINS_GOLD']}\n")
                f.write(f"      LLM reasoning:           {llm_reason}\n")
                f.write(f"      Deterministic reasoning: {row['ANSWER_CONTAINS_GOLD_REASON']}\n")
            f.write("\n")

    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
