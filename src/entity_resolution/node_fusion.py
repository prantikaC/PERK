import argparse
import json
import logging
import re

import networkx as nx
import pandas as pd

TITLE_WORDS = {
    'prof', 'prof.', 'dr', 'dr.', 'mr', 'mrs', 'ms', 'miss',
    'office', 'press', 'editorial', 'program', 'centre', 'center',
    'department', 'institute', 'university', 'college', 'school',
    'committee', 'board', 'chair', 'team', 'group', 'staff',
}


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


def _names_overlap(name_a: str, name_b: str) -> bool:
    """True if the two person name strings are likely the same person."""
    a, b = name_a.lower().strip(), name_b.lower().strip()
    if not a or not b or len(b) < 4:
        return False
    if a in b or b in a:
        return True
    words_a = {w for w in re.split(r'\s+', a) if len(w) >= 4 and w not in TITLE_WORDS}
    words_b = {w for w in re.split(r'\s+', b) if len(w) >= 4 and w not in TITLE_WORDS}
    return bool(words_a & words_b)


def patch_person_properties(df_ent: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Fill missing affiliation on Person nodes that have an email but no affiliation,
    by finding a complementary Person node (same partial name, has affiliation, no email).
    Operates entirely on the entities DataFrame — no Neo4j required.
    """
    persons = df_ent[df_ent['type'] == 'Person'].copy()
    if persons.empty:
        return df_ent, 0

    # Parse properties once
    parsed = {}
    for _, row in persons.iterrows():
        try:
            parsed[row['id']] = json.loads(row['properties'])
        except Exception:
            parsed[row['id']] = {}

    has_email = {eid: p for eid, p in parsed.items() if p.get('personEmail')}
    has_affil = {eid: p for eid, p in parsed.items()
                 if p.get('affiliation') and not p.get('personEmail')}

    patches = {}  # canonical_id → affiliation to add
    for eid_a, props_a in has_email.items():
        if props_a.get('affiliation'):
            continue  # already has affiliation
        name_a = props_a.get('personName', '')
        for eid_b, props_b in has_affil.items():
            name_b = props_b.get('personName', '')
            if _names_overlap(name_a, name_b):
                patches[eid_a] = props_b['affiliation']
                break

    if not patches:
        return df_ent, 0

    df_out = df_ent.copy()
    for idx, row in df_out.iterrows():
        if row['id'] in patches:
            try:
                props = json.loads(row['properties'])
                props['affiliation'] = patches[row['id']]
                df_out.at[idx, 'properties'] = json.dumps(props)
            except Exception:
                pass

    return df_out, len(patches)


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Node Fusion")
    parser.add_argument("--llm_resolved",    required=True, help="Input LLM resolved CSV")
    parser.add_argument("--raw_entities",    required=True, help="Input raw entities CSV")
    parser.add_argument("--raw_relations",   required=True, help="Input raw relations CSV")
    parser.add_argument("--fused_entities",  required=True, help="Output fused entities CSV")
    parser.add_argument("--fused_relations", required=True, help="Output fused relations CSV")
    parser.add_argument("--log", default="pipeline_step3.log", help="Log file path")
    args = parser.parse_args()

    setup_logger(args.log)
    logging.info("Starting Graph Node Fusion (Transitive Closure)...")

    df_res = pd.read_csv(args.llm_resolved)
    df_matches = df_res[df_res['llm_prediction'] == 'MATCH']
    logging.info(f"Loaded {len(df_matches)} verified matches for fusion.")

    G = nx.Graph()
    for _, r in df_matches.iterrows():
        G.add_edge(r['entity1_id'], r['entity2_id'])

    id_map = {}
    clusters_found = 0
    for cluster in nx.connected_components(G):
        clusters_found += 1
        c_list = sorted(list(cluster))
        canonical = c_list[0]
        for obs in c_list[1:]:
            id_map[obs] = canonical

    logging.info(f"Graph resolved into {clusters_found} distinct entity clusters.")
    logging.info(f"Identified {len(id_map)} redundant nodes to be merged into canonical IDs.")

    logging.info("Rewriting Relations (Edges)...")
    df_rel = pd.read_csv(args.raw_relations)
    initial_edges = len(df_rel)
    df_rel['start_id'] = df_rel['start_id'].apply(lambda x: id_map.get(x, x))
    df_rel['end_id']   = df_rel['end_id'].apply(lambda x: id_map.get(x, x))
    df_rel = df_rel[df_rel['start_id'] != df_rel['end_id']]
    df_rel = df_rel.drop_duplicates(subset=['start_id', 'end_id', 'relation', 'context'])
    final_edges = len(df_rel)
    df_rel.to_csv(args.fused_relations, index=False)

    logging.info("Rewriting Entities (Nodes)...")
    df_ent = pd.read_csv(args.raw_entities)
    initial_nodes = len(df_ent)
    df_ent.set_index('id', drop=False, inplace=True)

    properties_merged = 0
    for obs, can in id_map.items():
        if obs in df_ent.index and can in df_ent.index:
            try:
                obs_p = json.loads(df_ent.at[obs, 'properties'])
                can_p = json.loads(df_ent.at[can, 'properties'])
                df_ent.at[can, 'properties'] = json.dumps({**obs_p, **can_p})
                properties_merged += 1
            except Exception:
                pass

    df_ent = df_ent[~df_ent['id'].isin(id_map.keys())]
    final_nodes = len(df_ent)

    # Fill missing affiliation on fragmented Person nodes
    df_ent, persons_patched = patch_person_properties(df_ent)

    df_ent.to_csv(args.fused_entities, index=False)

    logging.info("--- Node Fusion Results ---")
    logging.info(f"Nodes before fusion:  {initial_nodes}")
    logging.info(f"Nodes after fusion:   {final_nodes} (Removed {initial_nodes - final_nodes})")
    logging.info(f"Properties merged:    {properties_merged}")
    logging.info(f"Edges before fusion:  {initial_edges}")
    logging.info(f"Edges after fusion:   {final_edges} (Consolidated {initial_edges - final_edges})")
    logging.info(f"Person nodes patched: {persons_patched} (affiliation filled from fragment)")
    logging.info(f"Output saved to '{args.fused_entities}' and '{args.fused_relations}'.")


if __name__ == "__main__":
    main()
