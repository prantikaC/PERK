# -*- coding: utf-8 -*-
"""
Applies acronym_cluster_audit.py's candidate pairs as an actual second
node-fusion round on top of already-fused entities/relations.

Reuses node_fusion.py's own merge machinery (canonical selection by
completeness, per-property longest-wins merge, NAME_KEYS abbreviation
collapse) rather than reimplementing it -- this is structurally the same
operation (connected components over verified-MATCH edges -> one canonical
per cluster), just running a second time with a different edge source.

Hard integrity gate before anything is written (per explicit supervisor
requirement): every entity id present in --fused_entities before this round
must be traceable afterward -- either still its own surviving row, or
listed in some surviving canonical's mergedFrom (including mergedFrom
entries carried forward from nodes absorbed this round). Any id that is
neither is "disappeared" -- rule-based proof of a bad merge, not something
inferred from confidence scores. Any cluster responsible for a disappearance
is excluded wholesale and its members restored to their pre-merge,
unmerged state (not a full-run rollback -- only the implicated cluster).

Dry-run by default, matching this repo's convention for every other script
that mutates the entity/relation CSVs (clean_kg.py, wipe_kg.py,
build_perk.py, the post-load repair scripts) -- pass --apply to write.
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from node_fusion import (  # noqa: E402
    NAME_KEYS, collapse_name_variants, completeness, merge_properties,
    setup_logger, union_sources,
)


def build_clusters(df_pairs, entity_completeness):
    """Connected components over the candidate-pairs edges -> one canonical
    id per cluster (most complete entity, ties by id, same rule
    node_fusion.py uses) plus the list of ids absorbed into it."""
    G = nx.Graph()
    for _, r in df_pairs.iterrows():
        G.add_edge(r["entity1_id"], r["entity2_id"])

    clusters = {}
    for component in nx.connected_components(G):
        c_list = sorted(component)
        canonical = min(c_list, key=lambda eid: (-entity_completeness.get(eid, 0), eid))
        obs_list = [eid for eid in c_list if eid != canonical]
        clusters[canonical] = obs_list
    return clusters


def apply_clusters(df_ent, df_rel, clusters):
    """Applies one set of clusters (canonical -> [absorbed ids]) to a COPY
    of df_ent/df_rel. Returns (new_df_ent, new_df_rel, id_map). Does not
    touch the caller's originals -- lets the integrity check inspect the
    result before anything is committed, and lets a failed check retry with
    a reduced cluster set without having double-mutated state."""
    df_ent = df_ent.copy()
    df_rel = df_rel.copy()

    id_map = {}
    for canonical, obs_list in clusters.items():
        for obs in obs_list:
            id_map[obs] = canonical

    df_rel["start_id"] = df_rel["start_id"].apply(lambda x: id_map.get(x, x))
    df_rel["end_id"] = df_rel["end_id"].apply(lambda x: id_map.get(x, x))
    df_rel = df_rel[df_rel["start_id"] != df_rel["end_id"]]
    df_rel["source"] = df_rel.groupby(
        ["start_id", "end_id", "relation", "context"]
    )["source"].transform(union_sources)
    df_rel = df_rel.drop_duplicates(subset=["start_id", "end_id", "relation", "context"])

    for obs, can in id_map.items():
        if obs not in df_ent.index or can not in df_ent.index:
            continue
        try:
            obs_p = json.loads(df_ent.at[obs, "properties"])
            can_p = json.loads(df_ent.at[can, "properties"])
            df_ent.at[can, "properties"] = json.dumps(merge_properties(obs_p, can_p))
        except Exception as e:
            logging.error(f"Property merge failed for {obs} -> {can}: {e}")

    # Provenance: union the canonical's EXISTING mergedFrom (from round 1)
    # with the ids absorbed this round, AND with those absorbed ids' own
    # prior mergedFrom entries -- an absorbed node that itself already
    # represented earlier raw mentions must carry that history forward, or
    # those raw ids become untraceable (exactly the "disappearance" the
    # integrity check below exists to catch).
    for canonical, obs_list in clusters.items():
        if canonical not in df_ent.index or not obs_list:
            continue
        try:
            props = json.loads(df_ent.at[canonical, "properties"])
        except Exception:
            props = {}
        prior_mf = props.get("mergedFrom", [])
        if not isinstance(prior_mf, list):
            prior_mf = []
        new_mf = list(prior_mf)
        for obs in obs_list:
            if obs not in new_mf:
                new_mf.append(obs)
            if obs in df_ent.index:
                try:
                    obs_props = json.loads(df_ent.at[obs, "properties"])
                    obs_prior_mf = obs_props.get("mergedFrom", [])
                    if isinstance(obs_prior_mf, list):
                        for sub_id in obs_prior_mf:
                            if sub_id not in new_mf:
                                new_mf.append(sub_id)
                except Exception:
                    pass
        props["mergedFrom"] = new_mf

        for key in NAME_KEYS:
            if isinstance(props.get(key), list):
                props[key] = collapse_name_variants(props[key])

        df_ent.at[canonical, "properties"] = json.dumps(props)

    df_ent = df_ent[~df_ent["id"].isin(id_map.keys())]
    return df_ent, df_rel, id_map


def find_disappeared(pre_merge_ids, df_ent_after):
    """Every id in pre_merge_ids must, in df_ent_after, either still be its
    own surviving row or appear in some surviving row's mergedFrom. Returns
    the set of ids that are neither -- i.e. genuinely disappeared."""
    accounted_for = set(df_ent_after["id"])
    for _, row in df_ent_after.iterrows():
        try:
            mf = json.loads(row["properties"]).get("mergedFrom", [])
        except Exception:
            mf = []
        if isinstance(mf, list):
            accounted_for.update(mf)
    return pre_merge_ids - accounted_for


def main():
    parser = argparse.ArgumentParser(description="Apply acronym_cluster_audit.py candidate pairs as a second node-fusion round")
    parser.add_argument("--fused_entities",  required=True, help="Current (post node_fusion.py) entities CSV")
    parser.add_argument("--fused_relations", required=True, help="Current (post node_fusion.py) relations CSV")
    parser.add_argument("--candidate_pairs", required=True, help="acronym_cluster_audit.py output CSV")
    parser.add_argument("--output_entities",  required=True, help="Output entities CSV (new file, input is never overwritten)")
    parser.add_argument("--output_relations", required=True, help="Output relations CSV (new file, input is never overwritten)")
    parser.add_argument("--apply", action="store_true",
                         help="Actually write --output_entities/--output_relations. Without this, "
                              "reports what WOULD happen and exits (dry run), matching this repo's "
                              "convention for every other entity/relation-mutating script.")
    parser.add_argument("--log", default="apply_acronym_merges.log")
    args = parser.parse_args()

    setup_logger(args.log)
    logging.info("======== APPLY ACRONYM MERGES STARTED ========")
    logging.info(f"Mode: {'APPLY (writing output)' if args.apply else 'DRY RUN (no files written)'}")

    df_ent = pd.read_csv(args.fused_entities)
    df_ent.set_index("id", drop=False, inplace=True)
    df_rel = pd.read_csv(args.fused_relations)
    df_pairs = pd.read_csv(args.candidate_pairs)

    pre_merge_ids = set(df_ent.index)
    initial_nodes, initial_edges = len(df_ent), len(df_rel)

    valid_ids = pre_merge_ids
    bad_mask = ~(df_pairs["entity1_id"].isin(valid_ids) & df_pairs["entity2_id"].isin(valid_ids))
    if bad_mask.any():
        bad = list(zip(df_pairs.loc[bad_mask, "entity1_id"], df_pairs.loc[bad_mask, "entity2_id"]))
        logging.warning(f"Dropping {bad_mask.sum()} candidate pair(s) referencing an id not present in "
                         f"--fused_entities (stale --candidate_pairs file?): {bad}")
    df_pairs = df_pairs[~bad_mask]

    entity_completeness = {}
    for eid in df_ent.index:
        try:
            entity_completeness[eid] = completeness(json.loads(df_ent.at[eid, "properties"]))
        except Exception:
            entity_completeness[eid] = 0

    clusters = build_clusters(df_pairs, entity_completeness)
    logging.info(f"{len(df_pairs)} candidate pairs -> {len(clusters)} proposed cluster(s), "
                 f"{sum(len(v) for v in clusters.values())} node(s) to be absorbed.")

    excluded_clusters = {}
    while True:
        new_ent, new_rel, id_map = apply_clusters(df_ent, df_rel, clusters)
        disappeared = find_disappeared(pre_merge_ids, new_ent)
        if not disappeared:
            break
        # Identify and exclude every cluster touching a disappeared id, then retry.
        bad_clusters = {
            can: obs_list for can, obs_list in clusters.items()
            if can in disappeared or any(o in disappeared for o in obs_list)
        }
        logging.error(f"INTEGRITY CHECK FAILED: {len(disappeared)} id(s) disappeared without a trace "
                      f"(neither a surviving row nor listed in any mergedFrom): {sorted(disappeared)}. "
                      f"Invalidating {len(bad_clusters)} cluster(s) responsible and restoring their "
                      f"members to pre-merge (unmerged) state: {bad_clusters}")
        excluded_clusters.update(bad_clusters)
        clusters = {can: obs for can, obs in clusters.items() if can not in bad_clusters}
        if not clusters:
            logging.info("No clusters remain after excluding all integrity-check failures.")
            new_ent, new_rel, id_map = df_ent.copy(), df_rel.copy(), {}
            break

    final_disappeared = find_disappeared(pre_merge_ids, new_ent)
    if final_disappeared:
        raise RuntimeError(f"Integrity check still failing after excluding all implicated clusters -- "
                            f"this should not happen: {sorted(final_disappeared)}")

    logging.info("--- Integrity check: PASSED (every pre-merge id is traceable in the output) ---")
    if excluded_clusters:
        logging.info(f"{len(excluded_clusters)} cluster(s) excluded due to the integrity check; "
                     f"{len(clusters)} cluster(s) applied successfully.")

    logging.info("--- Merge Results ---")
    logging.info(f"Nodes before: {initial_nodes} | Nodes after: {len(new_ent)} (removed {initial_nodes - len(new_ent)})")
    logging.info(f"Edges before: {initial_edges} | Edges after: {len(new_rel)} (consolidated {initial_edges - len(new_rel)})")
    logging.info(f"Clusters applied: {len(clusters)} | Nodes absorbed: {len(id_map)}")

    if args.apply:
        new_ent.to_csv(args.output_entities, index=False)
        new_rel.to_csv(args.output_relations, index=False)
        logging.info(f"Written: {args.output_entities}, {args.output_relations}")
    else:
        logging.info("DRY RUN -- no files written. Re-run with --apply to write "
                     f"{args.output_entities} / {args.output_relations}.")

    logging.info("======== APPLY ACRONYM MERGES COMPLETED ========")


if __name__ == "__main__":
    main()
