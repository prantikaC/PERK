# -*- coding: utf-8 -*-
"""
Materialize a direct Person-[:affiliation]->(Organization|Journal|Conference)
edge wherever one currently only exists transitively via
Person-[:memberOf]->Team-[:affiliation]->(...).

Why: header_signature_parser.py's own-signature extraction routes a person's
affiliation through their Team (e.g. "Department of Computer Science")
rather than straight to the Organization, while body-extraction mentions of
the same person by someone else often land a direct Person->Organization
edge instead. A real person can end up with only ONE of the two paths
populated (confirmed: Ananya Chatterjee's pn5 fragment has only the Team-
mediated path to "University of Oxford", nothing direct). A person is
affiliated with their team's organization by simple transitivity -- that's
a real fact, not just a query-time inference -- so it belongs in the graph
itself rather than requiring every future Cypher query to know to check
both hops.

This does NOT touch or merge the underlying duplicate Person fragments
(pn4/pn5/pn9-type splits) -- that's a separate entity-resolution problem.
This only adds the missing direct edge for whichever fragment already has
the Team-mediated path, so a query anchored on THAT fragment stops missing
data it should already be able to see.

Usage (dry-run by default):
    python materialize_transitive_affiliation.py --model gpt_v5
Apply:
    python materialize_transitive_affiliation.py --model gpt_v5 --apply
"""

import argparse
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)


def get_driver(prefix):
    prefix = prefix.upper()
    uri = os.getenv(f"{prefix}_NEO4J_URI")
    user = os.getenv(f"{prefix}_NEO4J_USERNAME")
    pwd = os.getenv(f"{prefix}_NEO4J_PASSWORD")
    if not all([uri, user, pwd]):
        raise ValueError(f"Missing {prefix}_NEO4J_URI/USERNAME/PASSWORD in .env")
    return GraphDatabase.driver(uri, auth=(user, pwd))


FIND_MISSING_QUERY = """
MATCH (p:Person)-[m:memberOf]->(t:Team)-[:affiliation]->(o)
WHERE NOT (p)-[:affiliation]->(o)
RETURN DISTINCT p.id AS person_id, p.personName AS person_name,
                t.id AS team_id, t.teamName AS team_name,
                labels(o) AS org_labels, o.id AS org_id,
                coalesce(o.orgName, o.journalTitle, o.confTitle) AS org_name,
                m.role AS role, m.date AS date
"""

# role/date are copied from the Person-[:memberOf]->Team edge this direct
# edge is derived from, not left null. That memberOf edge is where a
# person's role (e.g. "Research Assistant Professor") actually lives --
# nulling it here silently threw the fact away even though it was sitting
# right there in the same MATCH pattern. Confirmed real impact: "What is
# Soumya Banerjee's position at Old Dominion University?" (her Team's
# parent Organization) returned role=null from this materialized edge, even
# though her own memberOf edge to the Team correctly had role="Research
# Assistant Professor" the whole time.
APPLY_QUERY = """
MATCH (p:Person)-[m:memberOf]->(t:Team)-[:affiliation]->(o)
WHERE NOT (p)-[:affiliation]->(o)
MERGE (p)-[r:affiliation]->(o)
ON CREATE SET r.date = m.date, r.role = m.role,
              r.source = "materialize_transitive_affiliation.py (via " + t.id + ")"
RETURN count(DISTINCT r) AS created
"""


def main():
    parser = argparse.ArgumentParser(
        description="Materialize direct Person-affiliation->Org edges implied by "
                    "Person-memberOf->Team-affiliation->Org chains."
    )
    parser.add_argument("--model", required=True, help="Prefix used to look up env vars, e.g. 'gpt_v5'")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run only)")
    args = parser.parse_args()

    driver = get_driver(args.model)
    with driver.session() as session:
        missing = list(session.run(FIND_MISSING_QUERY))
        print(f"Found {len(missing)} Person->Team->Org chain(s) missing the direct affiliation edge:")
        for row in missing:
            print(f"  {row['person_name']} ({row['person_id']}) --memberOf--> "
                  f"{row['team_name']} ({row['team_id']}) --affiliation--> "
                  f"{row['org_name']} ({row['org_id']}, {row['org_labels']})")

        if not missing:
            print("Nothing to do.")
            return

        if args.apply:
            result = session.run(APPLY_QUERY).single()
            print(f"\nCreated {result['created']} new direct affiliation edge(s).")
        else:
            print(f"\n[DRY-RUN] Would create {len(missing)} new direct affiliation edge(s). "
                  f"Re-run with --apply to write them.")

    driver.close()


if __name__ == "__main__":
    main()
