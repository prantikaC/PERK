# -*- coding: utf-8 -*-
"""
Compare two models' raw extraction (entities + relations) against the golden
triples set (refined_golden_set_target.csv, 2372 triples across 243 emails),
then test whether the recall difference is statistically significant via
paired bootstrap resampling and an approximate randomization (permutation)
test -- the two significance tests standard in MT/NLP evaluation (Koehn 2004;
Noreen 1989) for comparing two systems on the same test set.

Per golden triple (subject_label/type, predicate, object_label/type,
email_id), a model "hits" it if its own raw extraction has a relation of
that predicate connecting a matching subject entity to a matching object
entity, sourced from the same email. Entity matching is exact
(type + case/whitespace/quote-insensitive text) first, falling back to
token-containment fuzzy matching for the five types where two independent
extractions of the same corpus should describe a mention almost identically
(Person, Conference, Journal, Paper, SubmissionID) -- same rationale as
check_resolution_correctness_neo4j.py. Task/Method/Metric/Dataset get no
fuzzy fallback: they're free-form paraphrases, so exact match is the only
safe signal and their recall will be a real (not an artifact) sensitivity
lever between models.

email_id in the golden file is anchored to whichever extraction the golden
set triples were originally written against -- confirmed to be openai_v2's
own Email entity ids (e.g. "e141"). This script maps email_id -> the
corpus's own mailNum (a real value parsed from the raw email header, model-
independent) via THAT anchor file once, then locates the matching mailNum
in each model's own extraction independently, so this works even if a
model's own internal id numbering doesn't line up with the golden file's.

Usage:
    python compare_extraction_recall.py \
        --golden ../../data/extraction_gold/refined_golden_set_target.csv \
        --golden_anchor_entities ../../data/entity_resolution/openai_v2/openai_v2_entities_final.csv \
        --model_a_name GPT --model_a_entities .../openai_v2_entities_final.csv --model_a_relations .../openai_v2_relations_final.csv \
        --model_b_name Qwen --model_b_entities .../qwen32b_v1_entities_final.csv --model_b_relations .../qwen32b_v1_relations_final.csv \
        --output per_triple_hits.csv \
        --n_resamples 10000
"""

import argparse
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

TYPE_LABEL_KEY = {
    "Email": "mailNum",
    "MailThread": "threadID",
    "EmailID": "eID",
    "Person": "personName",
    "Team": "teamName",
    "Organization": "orgName",
    "Paper": "paperTitle",
    "Conference": "confTitle",
    "Journal": "journalTitle",
    "SubmissionID": "identifier",
    "PaperStatus": "statusType",
    "Meeting": "meetAgenda",
    "Dataset": "datasetName",
    "Method": "methodName",
    "Task": "taskName",
    "Metric": "metricName",
}

FUZZY_TYPES = {"Person", "Conference", "Journal", "Paper", "SubmissionID"}

# The golden set's own predicate vocabulary (confirmed by inspecting
# refined_golden_set_target.csv's predicate column). Precision is only
# computed over predictions using one of these predicates, and only within
# emails the golden set actually annotated -- a predicted triple using a
# predicate or landing in an email outside this scope has no ground truth
# to be judged against, so counting it as a false positive would be unfair,
# not rigorous.
GOLD_PREDICATES = {
    "worksOn", "attends", "worksWith", "usedFor", "hasAuthor",
    "evaluates", "movesTo", "uses", "identifies", "inVenue",
}

QUOTE_CHARS = "'\"‘’“”"
HONORIFIC_RE = re.compile(r"^(?:dr|prof|mr|mrs|ms|miss)\.?\s+", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def norm(s):
    return str(s).strip().strip(QUOTE_CHARS).strip().lower()


def strip_honorific(s):
    return HONORIFIC_RE.sub("", s.strip())


def tokenize(s):
    s = strip_honorific(norm(s))
    s = re.sub(r"[^\w\s]", " ", s)
    return {t for t in s.split() if t}


def years_of(s):
    return set(YEAR_RE.findall(s))


def token_containment_match(etype, label_a, label_b):
    ta, tb = tokenize(label_a), tokenize(label_b)
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if not shorter or not shorter.issubset(longer):
        return False
    if etype == "Conference":
        ya, yb = years_of(label_a), years_of(label_b)
        if ya and yb and not (ya & yb):
            return False
    return True


def entity_matches(etype, label_a, label_b):
    """Same rule used for the gold-anchored recall check (hit()), applied in
    the reverse direction for precision: is a MODEL's own entity label the
    same real-world thing as a GOLD entity label. Exact match first, then
    the same token-containment fallback for the five stable-naming types."""
    if norm(label_a) == norm(label_b):
        return True
    if etype in FUZZY_TYPES:
        return token_containment_match(etype, label_a, label_b)
    return False


class ModelIndex:
    """Everything needed to check golden-triple hits against one model's raw extraction."""

    # A relation's `source` field is sometimes the resolved mailNum (e.g.
    # "MAILEX83T5QP"), sometimes a raw "email_<N>" placeholder that indexes
    # the SAME Email entity by its own id ("email_849" -> entity "e849") --
    # confirmed to hold for every sampled case. Both forms are resolved to
    # the one real mailNum here so source-matching doesn't silently miss
    # every relation using the placeholder form.
    EMAIL_PLACEHOLDER_RE = re.compile(r"^email_(\d+)$")

    def __init__(self, name, entities_path, relations_path):
        self.name = name
        self.label_index = defaultdict(list)   # (type, norm(label)) -> [raw ids]
        self.all_by_type = defaultdict(list)    # type -> [(raw id, raw label)]
        self.mailnum_to_id = {}                 # mailNum -> Email raw id
        self.id_to_mailnum = {}                 # Email raw id -> mailNum
        self.id_to_type_label = {}              # raw id -> (type, label), for precision lookups

        ent = pd.read_csv(entities_path)
        for _, row in ent.iterrows():
            etype = row["type"]
            try:
                props = json.loads(row["properties"])
            except Exception:
                continue
            if etype == "Email":
                mn = props.get("mailNum")
                rid = str(row["id"]).strip()
                if mn:
                    self.mailnum_to_id[mn] = rid
                    self.id_to_mailnum[rid] = mn
                continue
            key = TYPE_LABEL_KEY.get(etype)
            if not key:
                continue
            label = props.get(key)
            if label is None:
                continue
            rid = str(row["id"]).strip()
            self.label_index[(etype, norm(label))].append(rid)
            self.all_by_type[etype].append((rid, label))
            self.id_to_type_label[rid] = (etype, label)

        rel = pd.read_csv(relations_path)
        self.rel_index = defaultdict(list)      # predicate -> [(start_id, end_id, {mailnums})]
        for _, row in rel.iterrows():
            source = str(row.get("source", "") or "")
            mailnums = {self._resolve_source_token(s.strip())
                        for s in source.split(";") if s.strip()}
            mailnums.discard(None)
            self.rel_index[row["relation"]].append(
                (str(row["start_id"]).strip(), str(row["end_id"]).strip(), mailnums)
            )

    def _resolve_source_token(self, token):
        m = self.EMAIL_PLACEHOLDER_RE.match(token)
        if m:
            return self.id_to_mailnum.get(f"e{m.group(1)}")
        if token in self.id_to_mailnum:
            # Some pipelines (Gemma, Llama3) write the relation's source as
            # the Email entity's own raw id directly (e.g. "e643") instead
            # of either the real mailNum or the "email_N" placeholder.
            return self.id_to_mailnum[token]
        return token

    def candidate_ids(self, etype, label):
        if pd.isna(label):
            return set()
        ids = set(self.label_index.get((etype, norm(label)), []))
        if ids or etype not in FUZZY_TYPES:
            return ids
        return {
            rid for rid, rlabel in self.all_by_type.get(etype, [])
            if token_containment_match(etype, label, rlabel)
        }

    def hit(self, subject_type, subject_label, predicate, object_type, object_label, mailnum):
        if mailnum is None:
            return False
        subj_ids = self.candidate_ids(subject_type, subject_label)
        obj_ids = self.candidate_ids(object_type, object_label)
        if not subj_ids or not obj_ids:
            return False
        for start_id, end_id, mailnums in self.rel_index.get(predicate, []):
            if start_id in subj_ids and end_id in obj_ids and mailnum in mailnums:
                return True
        return False

    def predicted_triples_in_scope(self, golden_mailnums):
        """Yield (mailnum, predicate, subj_type, subj_label, obj_type, obj_label)
        for every relation this model predicted, restricted to (a) predicates
        the golden set covers and (b) emails the golden set actually
        annotated. Outside that scope there is no ground truth to judge a
        prediction against, so it is excluded rather than counted as a false
        positive by default."""
        for predicate, entries in self.rel_index.items():
            if predicate not in GOLD_PREDICATES:
                continue
            for start_id, end_id, mailnums in entries:
                scoped = mailnums & golden_mailnums
                if not scoped:
                    continue
                subj = self.id_to_type_label.get(start_id)
                obj = self.id_to_type_label.get(end_id)
                if not subj or not obj:
                    continue
                for mn in scoped:
                    yield mn, predicate, subj[0], subj[1], obj[0], obj[1]


def build_gold_index(golden_df, email_to_mailnum):
    """(mailnum, predicate) -> list of (subj_type, subj_label, obj_type, obj_label),
    for checking whether a model's own predicted triple matches some gold triple."""
    idx = defaultdict(list)
    for _, row in golden_df.iterrows():
        mn = email_to_mailnum.get(row["email_id"])
        if mn is None:
            continue
        idx[(mn, row["predicate"])].append(
            (row["subject_type"], row["subject_label"], row["object_type"], row["object_label"])
        )
    return idx


def compute_precision_stats(model, gold_index, golden_mailnums):
    """Per-mailnum (true-positive predictions, false-positive predictions)
    for one model, restricted to the golden set's scope (see
    predicted_triples_in_scope)."""
    tp_by_mail = defaultdict(int)
    fp_by_mail = defaultdict(int)
    for mn, predicate, s_type, s_label, o_type, o_label in model.predicted_triples_in_scope(golden_mailnums):
        gold_list = gold_index.get((mn, predicate), [])
        matched = any(
            g_s_type == s_type and g_o_type == o_type
            and entity_matches(s_type, g_s_label, s_label)
            and entity_matches(o_type, g_o_label, o_label)
            for g_s_type, g_s_label, g_o_type, g_o_label in gold_list
        )
        if matched:
            tp_by_mail[mn] += 1
        else:
            fp_by_mail[mn] += 1
    return tp_by_mail, fp_by_mail


def load_golden_mailnums(golden_df, anchor_entities_path):
    """golden email_id (anchored to one extraction's own Email ids) -> corpus mailNum."""
    ent = pd.read_csv(anchor_entities_path)
    id_to_mailnum = {}
    for _, row in ent.iterrows():
        if row["type"] != "Email":
            continue
        try:
            props = json.loads(row["properties"])
        except Exception:
            continue
        id_to_mailnum[str(row["id"]).strip()] = props.get("mailNum")
    return {eid: id_to_mailnum.get(str(eid).strip()) for eid in golden_df["email_id"].unique()}


def _per_email_stats(email_ids, a_hits, b_hits):
    """Group per-triple hit/miss into per-email (sum_a, sum_b, n_triples)
    totals. The unit that should be resampled/permuted is the email the
    triples were extracted from, not the triples themselves -- entities and
    relations within the same email are correlated (shared vocabulary,
    thread context), not independent, so resampling/permuting individual
    triples understates variance and overstates significance. Grouping
    first, then resampling/permuting whole emails, keeps each email's
    internal correlation structure intact in every trial."""
    df = pd.DataFrame({"email_id": email_ids, "a": a_hits, "b": b_hits})
    grouped = df.groupby("email_id").agg(sum_a=("a", "sum"), sum_b=("b", "sum"), n=("a", "size"))
    return grouped[["sum_a", "sum_b", "n"]].to_numpy(dtype=float)


def paired_bootstrap(email_ids, a_hits, b_hits, n_resamples=10000, seed=42):
    """Koehn (2004)-style paired bootstrap, resampling EMAILS (not individual
    triples) with replacement -- see _per_email_stats for why. Returns
    (mean_diff, 95% CI, p_value) where p_value is the fraction of resamples
    where B >= A (evidence AGAINST "A is better than B")."""
    rng = np.random.default_rng(seed)
    stats = _per_email_stats(email_ids, a_hits, b_hits)
    n_emails = len(stats)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n_emails, size=n_emails)
        sampled = stats[idx]
        total = sampled[:, 2].sum()
        diffs[i] = (sampled[:, 0].sum() - sampled[:, 1].sum()) / total
    mean_diff = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = float((diffs <= 0).mean())
    return mean_diff, (ci_low, ci_high), p_value


def approximate_randomization(email_ids, a_hits, b_hits, n_permutations=10000, seed=42):
    """Paired approximate randomization / permutation test (Noreen 1989),
    permuting the A/B assignment per EMAIL (not per triple) -- see
    _per_email_stats. Under H0, an email's whole set of per-model hit counts
    could equally well have been swapped between A and B; the recall
    difference is recomputed as a corpus-level (micro) ratio each trial, not
    a per-email average, since triple counts vary widely across emails.
    Returns (observed_diff, p_value), two-sided."""
    rng = np.random.default_rng(seed)
    stats = _per_email_stats(email_ids, a_hits, b_hits)
    n_emails = len(stats)
    total_triples = stats[:, 2].sum()
    observed_diff = (stats[:, 0].sum() - stats[:, 1].sum()) / total_triples
    count_ge = 0
    for _ in range(n_permutations):
        swap = rng.random(n_emails) < 0.5
        sum_a = np.where(swap, stats[:, 1], stats[:, 0]).sum()
        sum_b = np.where(swap, stats[:, 0], stats[:, 1]).sum()
        perm_diff = (sum_a - sum_b) / total_triples
        if abs(perm_diff) >= abs(observed_diff):
            count_ge += 1
    p_value = (count_ge + 1) / (n_permutations + 1)
    return observed_diff, p_value


def _f1_from_sums(tp_gold, fn, tp_pred, fp):
    """Precision uses a different numerator (tp_pred, matched PREDICTED
    triples) than recall (tp_gold, matched GOLD triples) -- these need not
    be equal, since one gold triple can be captured by several predicted
    triples or vice versa. Combining them via the standard F1 harmonic mean
    is still the correct way to get a single headline number from two
    separately-aligned P/R estimates."""
    recall = tp_gold / (tp_gold + fn) if (tp_gold + fn) > 0 else 0.0
    precision = tp_pred / (tp_pred + fp) if (tp_pred + fp) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def paired_bootstrap_f1(stats_a, stats_b, n_resamples=10000, seed=42):
    """Same email-level bootstrap as paired_bootstrap, but recomputing micro
    F1 (from summed tp_gold/fn/tp_pred/fp) each trial instead of a simple
    hit-ratio diff. stats_a/b: (n_emails, 4) arrays of [tp_gold, fn, tp_pred, fp]."""
    rng = np.random.default_rng(seed)
    n = len(stats_a)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        f1_a = _f1_from_sums(*stats_a[idx].sum(axis=0))
        f1_b = _f1_from_sums(*stats_b[idx].sum(axis=0))
        diffs[i] = f1_a - f1_b
    mean_diff = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = float((diffs <= 0).mean())
    return mean_diff, (ci_low, ci_high), p_value


def approximate_randomization_f1(stats_a, stats_b, n_permutations=10000, seed=42):
    """Same email-level AR as approximate_randomization, but on micro F1
    instead of a simple hit-ratio diff."""
    rng = np.random.default_rng(seed)
    n = len(stats_a)
    observed_diff = _f1_from_sums(*stats_a.sum(axis=0)) - _f1_from_sums(*stats_b.sum(axis=0))
    count_ge = 0
    for _ in range(n_permutations):
        swap = rng.random(n) < 0.5
        swapped_a = np.where(swap[:, None], stats_b, stats_a)
        swapped_b = np.where(swap[:, None], stats_a, stats_b)
        diff = _f1_from_sums(*swapped_a.sum(axis=0)) - _f1_from_sums(*swapped_b.sum(axis=0))
        if abs(diff) >= abs(observed_diff):
            count_ge += 1
    p_value = (count_ge + 1) / (n_permutations + 1)
    return observed_diff, p_value


def main():
    parser = argparse.ArgumentParser(
        description="Compare two extraction models' recall on the golden triples set, "
                    "with paired bootstrap and permutation significance tests."
    )
    parser.add_argument("--golden", required=True)
    parser.add_argument("--golden_anchor_entities", required=True,
                        help="Raw entities_final.csv of whichever model the golden file's "
                             "email_id column is anchored to (used only to resolve email_id -> mailNum)")
    parser.add_argument("--model_a_name", required=True)
    parser.add_argument("--model_a_entities", required=True)
    parser.add_argument("--model_a_relations", required=True)
    parser.add_argument("--model_b_name", required=True)
    parser.add_argument("--model_b_entities", required=True)
    parser.add_argument("--model_b_relations", required=True)
    parser.add_argument("--output", default=None, help="Optional CSV of per-triple hit/miss for both models")
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    golden = pd.read_csv(args.golden)
    email_to_mailnum = load_golden_mailnums(golden, args.golden_anchor_entities)

    model_a = ModelIndex(args.model_a_name, args.model_a_entities, args.model_a_relations)
    model_b = ModelIndex(args.model_b_name, args.model_b_entities, args.model_b_relations)

    rows = []
    for _, row in golden.iterrows():
        mailnum = email_to_mailnum.get(row["email_id"])
        a_hit = model_a.hit(row["subject_type"], row["subject_label"], row["predicate"],
                             row["object_type"], row["object_label"], mailnum)
        b_hit = model_b.hit(row["subject_type"], row["subject_label"], row["predicate"],
                             row["object_type"], row["object_label"], mailnum)
        rows.append({
            "email_id": row["email_id"],
            "mailnum": mailnum,
            "subject_type": row["subject_type"],
            "subject_label": row["subject_label"],
            "predicate": row["predicate"],
            "object_type": row["object_type"],
            "object_label": row["object_label"],
            f"{args.model_a_name}_hit": int(a_hit),
            f"{args.model_b_name}_hit": int(b_hit),
        })

    out_df = pd.DataFrame(rows)
    email_ids = out_df["email_id"].values
    a_hits = out_df[f"{args.model_a_name}_hit"].values
    b_hits = out_df[f"{args.model_b_name}_hit"].values

    recall_a, recall_b = a_hits.mean(), b_hits.mean()
    print(f"N golden triples: {len(out_df)} (across {out_df['email_id'].nunique()} emails)")
    print(f"{args.model_a_name} recall: {recall_a:.4f} ({a_hits.sum()}/{len(a_hits)})")
    print(f"{args.model_b_name} recall: {recall_b:.4f} ({b_hits.sum()}/{len(b_hits)})")
    print(f"Observed recall difference ({args.model_a_name} - {args.model_b_name}): {recall_a - recall_b:.4f}")
    print("(Both significance tests below resample/permute at the EMAIL level, not the "
          "individual-triple level -- see paired_bootstrap/approximate_randomization docstrings.)")

    print("\n--- Paired Bootstrap Resampling (email-level) ---")
    mean_diff, (ci_low, ci_high), boot_p = paired_bootstrap(
        email_ids, a_hits, b_hits, n_resamples=args.n_resamples, seed=args.seed
    )
    print(f"Mean resampled diff: {mean_diff:.4f}  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"p-value (P[{args.model_b_name} >= {args.model_a_name}] under resampling): {boot_p:.4f}")

    print("\n--- Approximate Randomization (Permutation Test, email-level) ---")
    observed_diff, perm_p = approximate_randomization(
        email_ids, a_hits, b_hits, n_permutations=args.n_resamples, seed=args.seed
    )
    print(f"Observed diff: {observed_diff:.4f}")
    print(f"Two-sided p-value: {perm_p:.4f}")

    # --- Precision and joint F1, over the same golden-set scope ---
    golden_mailnums = {mn for mn in email_to_mailnum.values() if mn}
    gold_index = build_gold_index(golden, email_to_mailnum)
    tp_pred_a, fp_a = compute_precision_stats(model_a, gold_index, golden_mailnums)
    tp_pred_b, fp_b = compute_precision_stats(model_b, gold_index, golden_mailnums)

    # Per-mailnum recall-side counts (tp_gold, fn), aligned to the same
    # mailnum key as the precision-side counts above.
    recall_by_mail = out_df.groupby("mailnum").agg(
        tp_gold_a=(f"{args.model_a_name}_hit", "sum"),
        tp_gold_b=(f"{args.model_b_name}_hit", "sum"),
        n=(f"{args.model_a_name}_hit", "size"),
    )

    mail_list = sorted(golden_mailnums)
    stats_a = np.array([
        [recall_by_mail.loc[mn, "tp_gold_a"] if mn in recall_by_mail.index else 0,
         (recall_by_mail.loc[mn, "n"] - recall_by_mail.loc[mn, "tp_gold_a"]) if mn in recall_by_mail.index else 0,
         tp_pred_a.get(mn, 0), fp_a.get(mn, 0)]
        for mn in mail_list
    ], dtype=float)
    stats_b = np.array([
        [recall_by_mail.loc[mn, "tp_gold_b"] if mn in recall_by_mail.index else 0,
         (recall_by_mail.loc[mn, "n"] - recall_by_mail.loc[mn, "tp_gold_b"]) if mn in recall_by_mail.index else 0,
         tp_pred_b.get(mn, 0), fp_b.get(mn, 0)]
        for mn in mail_list
    ], dtype=float)

    def summarize(stats):
        tp_gold, fn, tp_pred, fp = stats.sum(axis=0)
        recall = tp_gold / (tp_gold + fn) if (tp_gold + fn) else 0.0
        precision = tp_pred / (tp_pred + fp) if (tp_pred + fp) else 0.0
        f1 = _f1_from_sums(tp_gold, fn, tp_pred, fp)
        return precision, recall, f1, tp_pred, fp

    prec_a, rec_a, f1_a, tp_pred_total_a, fp_total_a = summarize(stats_a)
    prec_b, rec_b, f1_b, tp_pred_total_b, fp_total_b = summarize(stats_b)

    print(f"\n--- Precision / Joint F1 (scope: {len(golden_mailnums)} golden emails, "
          f"predicates {sorted(GOLD_PREDICATES)}) ---")
    print(f"{args.model_a_name}: precision {prec_a:.4f} ({int(tp_pred_total_a)}/{int(tp_pred_total_a + fp_total_a)}), "
          f"recall {rec_a:.4f}, F1 {f1_a:.4f}")
    print(f"{args.model_b_name}: precision {prec_b:.4f} ({int(tp_pred_total_b)}/{int(tp_pred_total_b + fp_total_b)}), "
          f"recall {rec_b:.4f}, F1 {f1_b:.4f}")

    print("\n--- Paired Bootstrap Resampling on F1 (email-level) ---")
    mean_diff_f1, (ci_low_f1, ci_high_f1), boot_p_f1 = paired_bootstrap_f1(
        stats_a, stats_b, n_resamples=args.n_resamples, seed=args.seed
    )
    print(f"Mean resampled F1 diff: {mean_diff_f1:.4f}  95% CI: [{ci_low_f1:.4f}, {ci_high_f1:.4f}]")
    print(f"p-value (P[{args.model_b_name} >= {args.model_a_name}] under resampling): {boot_p_f1:.4f}")

    print("\n--- Approximate Randomization on F1 (email-level) ---")
    observed_diff_f1, perm_p_f1 = approximate_randomization_f1(
        stats_a, stats_b, n_permutations=args.n_resamples, seed=args.seed
    )
    print(f"Observed F1 diff: {observed_diff_f1:.4f}")
    print(f"Two-sided p-value: {perm_p_f1:.4f}")

    if args.output:
        out_df.to_csv(args.output, index=False)
        print(f"\nSaved per-triple hit/miss to {args.output}")


if __name__ == "__main__":
    main()
