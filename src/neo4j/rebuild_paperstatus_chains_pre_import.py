# -*- coding: utf-8 -*-
"""
Rebuild clean, per-submission PaperStatus movesTo chains BEFORE graph
construction, operating on the fused entities/relations CSVs (the same files
clean_kg.py takes as input) instead of a live Neo4j graph.

Root cause: the raw extraction pipeline assigns PaperStatus a small, reused
set of local IDs per email (e.g. "ps7" always means "the Submitted status"),
correct behavior for stable real-world entities like Person/Dataset but wrong
for PaperStatus, where every occurrence is a distinct event in one specific
submission's lifecycle. Once aggregated across ~1000 emails, unrelated
submissions' status events collapse onto the same node. This corrupts "final
status" / "status history" questions, since walking a submission's chain can
wander into a different submission's real history.

Fix: reconstruct each submission's chain from the RAW pre-fusion per-email
extraction files (entity_extractions/, relation_extractions/), where local
IDs are still unambiguous within a single email. For each SubmissionID
(matched by its globally-unique `identifier` string), collect every raw
PaperStatus mention plus any same-email PaperStatus->PaperStatus ordering
edges, sort chronologically (own statusDate, falling back to the source
email's number when no date exists), and deduplicate consecutive repeats of
the same status label. Then splice the result into entities/relations CSVs
in place of the entangled PaperStatus subgraph, before clean_kg.py/
prepare_import.py/build_perk.py ever run.

Usage (dry-run by default):
    python rebuild_paperstatus_chains_pre_import.py \
        --raw-dir  src/entity_extractions_dir \
        --entities_in entities_fused.csv --relations_in relations_fused.csv \
        --entities_out entities_fixed.csv --relations_out relations_fixed.csv
Apply:
    ... --apply
"""

import argparse
import glob
import json
import os
import re
from datetime import date

import pandas as pd
from dateutil import parser as dateparser


def parse_date_loose(s):
    if not s or not isinstance(s, str):
        return None
    try:
        d = dateparser.parse(s, default=date(1, 1, 1), fuzzy=True)
        if d.year == 1:
            return None
        return d.date()
    except Exception:
        return None


def reconstruct_chains(raw_dir):
    """Returns {submission_identifier: [ {statusType, statusDate, context,
    source} , ... ]} in chronological order, deduplicated, built from raw
    per-email extractions -- identical logic to the live-graph version."""
    ent_dir = os.path.join(raw_dir, "entity_extractions")
    rel_dir = os.path.join(raw_dir, "relation_extractions")
    ent_files = sorted(glob.glob(os.path.join(ent_dir, "entities_email*.csv")),
                        key=lambda p: int(re.search(r'email(\d+)', p).group(1)))

    sub_events = {}

    for ent_f in ent_files:
        n = int(re.search(r'email(\d+)', ent_f).group(1))
        rel_f = os.path.join(rel_dir, f"relations_email{n}.csv")
        try:
            ent = pd.read_csv(ent_f)
            rel = pd.read_csv(rel_f)
        except Exception:
            continue

        subs, statuses = {}, {}
        mail_date = None
        for _, r in ent.iterrows():
            try:
                props = json.loads(r['properties'])
            except Exception:
                props = {}
            if r['type'] == 'SubmissionID':
                subs[r['id']] = props.get('identifier')
            elif r['type'] == 'PaperStatus':
                statuses[r['id']] = (props.get('statusType'), props.get('statusDate'))
            elif r['type'] == 'Email':
                mail_date = parse_date_loose(props.get('mailDate'))

        if not subs or not statuses or 'relation' not in rel.columns:
            continue

        mv = rel[rel['relation'] == 'movesTo']
        sub_to_ps, ps_chain_next, edge_ctx = {}, {}, {}
        for _, r in mv.iterrows():
            s, e = str(r['start_id']), str(r['end_id'])
            edge_ctx[(s, e)] = (r.get('context'), r.get('source'))
            if s in subs and e in statuses:
                sub_to_ps.setdefault(s, set()).add(e)
            elif s in statuses and e in statuses:
                ps_chain_next[s] = e

        for s_local, sub_id in subs.items():
            linked = sub_to_ps.get(s_local, set())
            if not linked:
                continue
            targets = {v for k, v in ps_chain_next.items() if k in linked and v in linked}
            heads = [x for x in linked if x not in targets]
            visited = set()
            for head in heads:
                pos, cur, prev = 0, head, None
                while cur is not None and cur not in visited:
                    visited.add(cur)
                    st_type, st_date = statuses.get(cur, (None, None))
                    edge_key = (s_local, cur) if prev is None else (prev, cur)
                    ctx, src = edge_ctx.get(edge_key, (None, None))
                    sub_events.setdefault(sub_id, []).append({
                        "statusType": st_type, "statusDate": parse_date_loose(st_date),
                        "mailDate": mail_date, "email_n": n, "chain_pos": pos,
                        "context": ctx, "source": src,
                    })
                    nxt = ps_chain_next.get(cur)
                    prev = cur
                    cur = nxt if nxt in linked else None
                    pos += 1
            for x in linked - visited:
                st_type, st_date = statuses.get(x, (None, None))
                ctx, src = edge_ctx.get((s_local, x), (None, None))
                sub_events.setdefault(sub_id, []).append({
                    "statusType": st_type, "statusDate": parse_date_loose(st_date),
                    "mailDate": mail_date, "email_n": n, "chain_pos": 0,
                    "context": ctx, "source": src,
                })

    def sort_key(ev):
        eff_date = ev["statusDate"] or ev["mailDate"] or date(9999, 1, 1)
        return (eff_date, ev["email_n"], ev["chain_pos"])

    result = {}
    for sub_id, events in sub_events.items():
        events = sorted(events, key=sort_key)
        clean = []
        for ev in events:
            if clean and clean[-1]["statusType"] == ev["statusType"]:
                if clean[-1]["statusDate"] is None and ev["statusDate"] is not None:
                    clean[-1] = ev
                continue
            clean.append(ev)
        result[sub_id] = clean
    return result


def apply_to_dataframes(df_ent, df_rel, chains):
    """Replace the entangled PaperStatus subgraph in df_ent/df_rel with the
    reconstructed chains, for submissions we can rebuild. Submissions with no
    reconstructable raw chain are left untouched."""
    id_to_identifier = {}
    for _, r in df_ent[df_ent['type'] == 'SubmissionID'].iterrows():
        try:
            props = json.loads(r['properties'])
        except Exception:
            props = {}
        ident = props.get('identifier')
        if ident:
            id_to_identifier[r['id']] = ident
    identifier_to_id = {v: k for k, v in id_to_identifier.items()}

    rebuildable_identifiers = set(chains.keys()) & set(identifier_to_id.keys())

    # Old PaperStatus nodes reachable (via movesTo, any depth) from a
    # rebuildable SubmissionID -- find via a simple forward BFS over movesTo.
    movesto = df_rel[df_rel['relation'] == 'movesTo']
    adj = {}
    for _, r in movesto.iterrows():
        adj.setdefault(r['start_id'], []).append(r['end_id'])

    old_ps_ids = set()
    frontier = [identifier_to_id[ident] for ident in rebuildable_identifiers]
    seen = set(frontier)
    while frontier:
        nxt = []
        for node in frontier:
            for child in adj.get(node, []):
                ps_type = df_ent.loc[df_ent['id'] == child, 'type']
                if not ps_type.empty and ps_type.iloc[0] == 'PaperStatus' and child not in seen:
                    old_ps_ids.add(child)
                    seen.add(child)
                    nxt.append(child)
        frontier = nxt

    # Drop old PaperStatus entities + any relation touching them.
    df_ent_new = df_ent[~df_ent['id'].isin(old_ps_ids)].copy()
    df_rel_new = df_rel[
        ~df_rel['start_id'].isin(old_ps_ids) & ~df_rel['end_id'].isin(old_ps_ids)
    ].copy()

    # Recreate clean chains with fresh globally-unique IDs.
    new_ent_rows, new_rel_rows = [], []
    for sub_id in rebuildable_identifiers:
        events = chains[sub_id]
        if not events:
            continue
        sub_local_id = identifier_to_id[sub_id]
        prev_id = None
        for i, ev in enumerate(events):
            new_id = f"ps_{sub_id}_{i}"
            props = {"statusType": ev["statusType"]}
            if ev["statusDate"]:
                props["statusDate"] = ev["statusDate"].isoformat()
            new_ent_rows.append({"id": new_id, "type": "PaperStatus", "properties": json.dumps(props)})

            edge_row = {
                "start_id": sub_local_id if prev_id is None else prev_id,
                "end_id": new_id,
                "relation": "movesTo",
                "context": ev.get("context"),
                "source": ev.get("source"),
            }
            new_rel_rows.append(edge_row)
            prev_id = new_id

    if new_ent_rows:
        df_ent_new = pd.concat([df_ent_new, pd.DataFrame(new_ent_rows)], ignore_index=True)
    if new_rel_rows:
        # Match df_rel's existing column set (role/date columns if present).
        new_rel_df = pd.DataFrame(new_rel_rows)
        for col in df_rel_new.columns:
            if col not in new_rel_df.columns:
                new_rel_df[col] = None
        new_rel_df = new_rel_df[df_rel_new.columns]
        df_rel_new = pd.concat([df_rel_new, new_rel_df], ignore_index=True)

    return df_ent_new, df_rel_new, rebuildable_identifiers, old_ps_ids


def main():
    p = argparse.ArgumentParser(
        description="Rebuild clean per-submission PaperStatus chains before graph construction."
    )
    p.add_argument("--raw-dir", required=True,
                    help="Dir containing entity_extractions/ and relation_extractions/ "
                         "subdirs of raw per-email extraction CSVs")
    p.add_argument("--entities_in", required=True)
    p.add_argument("--relations_in", required=True)
    p.add_argument("--entities_out", required=True)
    p.add_argument("--relations_out", required=True)
    p.add_argument("--apply", action="store_true", help="Write output (default: dry-run report only)")
    args = p.parse_args()

    chains = reconstruct_chains(args.raw_dir)
    print(f"Reconstructed chains for {len(chains)} submissions from raw data.\n")

    df_ent = pd.read_csv(args.entities_in)
    df_rel = pd.read_csv(args.relations_in)

    df_ent_new, df_rel_new, rebuilt, old_ps_ids = apply_to_dataframes(df_ent, df_rel, chains)

    old_ps_count = (df_ent['type'] == 'PaperStatus').sum()
    new_ps_count = (df_ent_new['type'] == 'PaperStatus').sum()
    print(f"Rebuilt {len(rebuilt)} submissions' chains "
          f"({len(old_ps_ids)} old PaperStatus nodes replaced).")
    print(f"PaperStatus count: {old_ps_count} -> {new_ps_count}")

    sample = list(rebuilt)[:3]
    for sub_id in sample:
        steps = " -> ".join(ev["statusType"] or "?" for ev in chains[sub_id])
        print(f"  {sub_id}: {steps}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write entities_out/relations_out.")
        return

    df_ent_new.to_csv(args.entities_out, index=False)
    df_rel_new.to_csv(args.relations_out, index=False)
    print(f"\n[APPLY] Wrote {args.entities_out} and {args.relations_out}")


if __name__ == "__main__":
    main()
