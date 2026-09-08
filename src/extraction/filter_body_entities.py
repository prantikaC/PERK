# -*- coding: utf-8 -*-
"""
Filter the body-extraction pipeline's entities_final.csv / relations_final.csv
down to BODY-only content, dropping everything header_signature_parser.py now
owns instead (Email, MailThread, and header-known Person/sentBy/receivedBy/
partOf).

kg_extraction_pipeline.py tags every header-derived relation (partOf,
sentBy, receivedBy) with source == "header" literally (see HEADER_EVIDENCE /
extract_header_entities) -- body-derived relations instead carry the
source email's own entity id (e.g. "e1") and an actual quoted context. That
tag is the ground truth this script filters relations on; no heuristics
needed there.

Person entities are the tricky case, because the OLD pipeline's own header
parsing already minted a Person node (with a personEmail property) for every
From/To/Cc participant -- exactly the set header_signature_parser.py now
produces independently, and typically far more accurately (proper Team vs
Person classification, signature-derived role/affiliation, etc). Simply
"keep if referenced by a body relation" is not enough: some of these
header-known Persons ALSO carry substantial body content (e.g. dozens of
worksOn/worksWith/attends edges), and dropping the entity would silently
strand or destroy that content.

So for any old-pipeline Person that has a personEmail, this script looks
that address up against header_signature_parser.py's own EmailID ->
hasOwner -> (Person|Team) mapping (--header_relations). If it matches:
  - every body relation touching that old Person id is REDIRECTED onto the
    header entity's own raw id (prepare_er_input.py keeps header ids
    untouched and only renumbers colliding body ids around them, so no
    tagging/prefixing is needed here), and
  - the now-redundant old Person entity is dropped.
The header owner can be a Team, not just a Person -- e.g. an address like
"acl2021@conferences.aclweb.org" is really an office/committee, and the old
pipeline's blanket "From/To/Cc participant -> Person" rule got that wrong.
Redirecting onto the Team id is still correct: it fixes the entity identity,
and if that leaves a relation type whose ontology domain requires Person
(worksOn, worksWith, hasAuthor, attends, ...) now pointing at a Team, that
triplet is exactly what prepare_er_input.py's ontology-violation filter is
for -- it gets dropped downstream, which is the right outcome for a
genuinely mistaken relation, not something this script needs to guard
against itself.

A Person with no personEmail match falls back to a second, still-deterministic
check before giving up: if the Person's name (salutation stripped) exactly
matches a header Person's full name, AND this Person's own "affiliation"
property names the same Organization that header Person already has an
affiliation relation to, they're the same real person under a different
address (e.g. Ramesh Bhatia mentioned in a body signature block under an
address the header script never saw, but with matching name + matching
institution) -- redirected exactly like the personEmail case. If the body
Person's own recorded name is JUST a bare first name (e.g. "Ramesh", no
surname at all -- not a full name that merely fails to match, such as a
typo'd surname), it's matched against header Persons' first names instead,
still gated on the same org match. Org names are matched as an exact,
case-insensitive string, not fuzzy -- a near-miss like "Stanford" vs
"Stanford University" is left alone rather than guessed at, and picked up
later by FAISS/LLM resolution instead, which is allowed to be fuzzy because
it's reviewed, not auto-merged.

A A third, still-deterministic check applies when neither of those match:
every non-header relation in relations_final.csv already names, in its own
'source' column, the exact Email entity it was extracted from -- so for any
body Person, every mailNum they're actually tied to (across all of their
relations, not just one) can be read directly off that ground truth, no
guessing needed. If any of those emails' own sender or recipient (per
header_relations.csv's sentBy/receivedBy, resolved through hasOwner) has
the exact same name -- or, if the body Person's own recorded name is just a
bare first name, the same first name -- they're almost certainly the same
person being referred to by name within an email whose participant list
already establishes who that name refers to -- redirected the same way.
This is narrower than a bare "same name anywhere" merge (which would be
unsafe -- two different real people can share a name) since it's anchored
to specific emails' own participant lists, not the whole corpus.

A Person matched by none of the above falls back to the original rule: kept
only if referenced by a surviving (non-header, non-redirected) body
relation.

A surviving Person can still be a flat-out type error from the old
pipeline's own LLM extraction -- e.g. personName "press office at IACS" or
"Indian National Digital Library Foundation" are institutions, not people.
This matters more than a cosmetic label: faiss_blocking.py buckets entities
by their declared type before comparing anything, so a Person-mistyped
institution can never be matched against real Team/Organization entities
during resolution, no matter how similar the names are -- it has to be
fixed at the type level, here, or not at all. Any surviving Person whose
name contains an institutional keyword (the same keyword list
header_signature_parser.py uses for its own Person/Team classification,
plus "foundation"/"library") is retyped to Team, with personName renamed to
teamName; whatever relations already pointed at that id keep pointing at it
unchanged. If that leaves a relation type whose ontology domain requires
Person (worksOn, worksWith, hasAuthor, attends, ...) now pointing at a
Team, that's the same ontology-violation cleanup prepare_er_input.py
already does for the redirect case above -- correct to drop, not something
this script needs to guard against.

A surviving Person that's genuinely a person also gets its personName
salutation-stripped (matching header_signature_parser.py's own rule that
personName never stores an honorific) -- but only when what's left still
has a first and last name; a bare "Dr. Bradley" or "Dr. K" keeps its title
rather than being reduced to a single, less identifying word.

Surviving Person entities can also still carry an "affiliation" property --
a leftover from the pre-ontology-change extraction prompt, which still asks
the LLM to fill Person.affiliation as a free-text string even though the
current PERKOnto.json no longer allows it (affiliation is a relation to
Organization/Journal/Conference now, not a Person property). Left alone,
clean_kg.py would just silently strip it at the very end and the fact would
be lost. Instead, this script converts it here: the string is split on
commas, the LAST segment becomes a new Organization entity's orgName (e.g.
"Professor, Department of Computer Science, Indian Association for the
Cultivation of Science" -> orgName "Indian Association for the Cultivation
of Science"), a leading extra segment (if any) is folded into the new
affiliation relation's context as the person's role, and the raw
"affiliation" key is removed from the Person's own properties. Before
minting a new Organization for that name, it's checked against
header_signature_parser.py's own Organization set (--header_entities) --
header already creates Organization nodes from signature chains, so e.g.
"Stanford University" or "University of Oxford" showing up here would
otherwise silently duplicate an Organization that already exists on the
header side. Only a name with no header match gets a new body-side "og" id;
prepare_er_input.py's collision renumbering keeps those disjoint from
header-side Organization ids.

Entities are handled per type:
  - Email, MailThread            : ALWAYS dropped -- these types are only
                                    ever produced by header parsing, never
                                    by body extraction.
  - Person                       : dropped and redirected if personEmail
                                    matches a header EmailID (see above);
                                    otherwise kept only if referenced by a
                                    surviving body relation.
  - everything else (Paper, Dataset, Method, Task, Metric, Meeting,
    SubmissionID, PaperStatus, Conference, Journal, PaperBib)
                                  : KEPT unconditionally -- these types are
                                    exclusively produced by body extraction.

Usage:
    python filter_body_entities.py \
        --entities_csv  <run>/final_outputs/entities_final.csv \
        --relations_csv <run>/final_outputs/relations_final.csv \
        --header_entities  data/header_extractions/openai_v2/header_entities.csv \
        --header_relations data/header_extractions/openai_v2/header_relations.csv \
        --output_dir    data/body_extractions/openai_v2/
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

HEADER_ONLY_ENTITY_TYPES = {"Email", "MailThread"}
PERSON_TITLE_RE = re.compile(r'^(Dr|Prof|Professor|Mr|Mrs|Ms|Miss)\.?\s+', re.IGNORECASE)
TEAM_KEYWORD_RE = re.compile(
    r'\b(committee|editorial|office|board|chairs?|organi[sz]ing|organizers?|team|group|'
    r'secretariat|desk|department|division|submissions?|reviewers?|panel|council|tutorials?|'
    r'foundation|library)\b', re.IGNORECASE
)


def load_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def normalize_person_name(name):
    return PERSON_TITLE_RE.sub('', name or '').strip().lower()


def strip_salutation_if_full_name(name):
    """Strip a leading honorific only when what's left still has a first
    and last name (>=2 tokens) -- a bare "Dr. Bradley" or "Dr. K" keeps its
    honorific rather than being reduced to a single, less identifying word."""
    stripped = PERSON_TITLE_RE.sub('', name or '').strip()
    return stripped if len(stripped.split()) >= 2 else name


def looks_like_team_name(name):
    return bool(TEAM_KEYWORD_RE.search(name or ''))


def build_address_to_owner(header_relations):
    """address (lowercased eID) -> (owner_id, owner_type, owner_label)."""
    address_to_owner = {}
    for row in header_relations:
        if row['relation'] != 'hasOwner':
            continue
        address = (row.get('start_label') or '').lower().strip()
        if address:
            address_to_owner[address] = (row['end_id'], row['end_type'], row['end_label'])
    return address_to_owner


def build_header_person_orgs(header_entities, header_relations):
    """Returns (by_full_name, by_first_name), each normalized-name ->
    [(person_id, {lowercased Organization names})]."""
    person_names = {}
    for row in header_entities:
        if row['type'] == 'Person':
            try:
                name = json.loads(row['properties']).get('personName')
            except Exception:
                name = None
            if name:
                person_names[row['id']] = name

    person_orgs = defaultdict(set)
    for row in header_relations:
        if row['relation'] == 'affiliation' and row['start_type'] == 'Person' and row['end_type'] == 'Organization':
            person_orgs[row['start_id']].add(row['end_label'].lower().strip())

    by_full_name, by_first_name = defaultdict(list), defaultdict(list)
    for pid, name in person_names.items():
        norm = normalize_person_name(name)
        by_full_name[norm].append((pid, person_orgs[pid]))
        first = norm.split()[0] if norm else None
        if first:
            by_first_name[first].append((pid, person_orgs[pid]))
    return by_full_name, by_first_name


def build_header_org_by_name(header_entities):
    """lowercased header Organization orgName -> its header entity id."""
    by_name = {}
    for row in header_entities:
        if row['type'] == 'Organization':
            try:
                name = json.loads(row['properties']).get('orgName')
            except Exception:
                name = None
            if name:
                by_name[name.lower().strip()] = row['id']
    return by_name


def find_header_owner_by_name_and_org(norm_name, body_org, by_full_name, by_first_name):
    for owner_id, org_names in by_full_name.get(norm_name, []):
        if body_org in org_names:
            return owner_id
    if norm_name and ' ' not in norm_name:
        # body Person recorded only a bare first name -- match against header
        # Persons' first names too, still gated on the org match.
        for owner_id, org_names in by_first_name.get(norm_name, []):
            if body_org in org_names:
                return owner_id
    return None


def build_email_id_to_mailnum(entity_rows):
    email_id_to_mailnum = {}
    for row in entity_rows:
        if row['type'] == 'Email':
            try:
                email_id_to_mailnum[row['id']] = json.loads(row['properties']).get('mailNum')
            except Exception:
                pass
    return email_id_to_mailnum


def build_person_to_source_mailnums(email_id_to_mailnum, relation_rows):
    """Person id -> set of mailNums, taken from the 'source' column of that
    Person's own non-header relations (each body relation's source is the
    exact Email entity id it was extracted from -- ground truth, not a
    nearest-preceding-row guess). A Person can carry relations sourced from
    several different emails, so this returns every mailNum they're actually
    tied to, not just one."""
    person_to_mailnums = defaultdict(set)
    for row in relation_rows:
        source = row.get('source')
        mailnum = email_id_to_mailnum.get(source) if source else None
        if not mailnum:
            continue
        person_to_mailnums[row['start_id']].add(mailnum)
        person_to_mailnums[row['end_id']].add(mailnum)
    return person_to_mailnums


def build_mailnum_to_header_persons(header_relations, address_to_owner):
    """mailNum -> [(header Person id, normalized name)] for that email's
    sentBy/receivedBy participants who resolve to a header Person owner."""
    mailnum_to_persons = defaultdict(list)
    for row in header_relations:
        if row['relation'] not in ('sentBy', 'receivedBy'):
            continue
        mailnum = row.get('start_label')
        address = (row.get('end_label') or '').lower().strip()
        owner = address_to_owner.get(address)
        if mailnum and owner and owner[1] == 'Person':
            owner_id, _owner_type, owner_label = owner
            mailnum_to_persons[mailnum].append((owner_id, normalize_person_name(owner_label)))
    return mailnum_to_persons


def main():
    parser = argparse.ArgumentParser(
        description="Filter entities_final.csv/relations_final.csv to body-only content, "
                    "redirecting header-known Persons onto their header EmailID owner."
    )
    parser.add_argument("--entities_csv", required=True)
    parser.add_argument("--relations_csv", required=True)
    parser.add_argument("--header_entities", required=True)
    parser.add_argument("--header_relations", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    entity_rows = load_rows(args.entities_csv)
    relation_rows = load_rows(args.relations_csv)
    header_entities = load_rows(args.header_entities)
    header_relations = load_rows(args.header_relations)

    address_to_owner = build_address_to_owner(header_relations)
    header_by_full_name, header_by_first_name = build_header_person_orgs(header_entities, header_relations)
    header_org_by_name = build_header_org_by_name(header_entities)
    email_id_to_mailnum = build_email_id_to_mailnum(entity_rows)
    person_to_source_mailnums = build_person_to_source_mailnums(email_id_to_mailnum, relation_rows)
    mailnum_to_header_persons = build_mailnum_to_header_persons(header_relations, address_to_owner)

    redirect_map = {}
    redirect_targets = Counter()
    name_affil_matches = 0
    context_matches = 0
    for row in entity_rows:
        if row['type'] != 'Person':
            continue
        try:
            props = json.loads(row['properties'])
        except Exception:
            props = {}

        email = (props.get('personEmail') or '').lower().strip()
        if email and email in address_to_owner:
            owner_id, owner_type, _owner_label = address_to_owner[email]
            redirect_map[row['id']] = owner_id
            redirect_targets[owner_type] += 1
            continue

        name = props.get('personName')
        norm_name = normalize_person_name(name) if name else None

        affiliation = props.get('affiliation')
        if norm_name and affiliation:
            body_org = affiliation.split(',')[-1].strip().lower()
            owner_id = find_header_owner_by_name_and_org(
                norm_name, body_org, header_by_full_name, header_by_first_name
            ) if body_org else None
            if owner_id:
                redirect_map[row['id']] = owner_id
                redirect_targets['Person'] += 1
                name_affil_matches += 1
                continue

        if norm_name:
            bare_first_name = ' ' not in norm_name
            matched = False
            for mailnum in person_to_source_mailnums.get(row['id'], ()):
                for cand_id, cand_norm_name in mailnum_to_header_persons.get(mailnum, []):
                    same_name = cand_norm_name == norm_name
                    same_first_name = bare_first_name and cand_norm_name.split()[:1] == [norm_name]
                    if same_name or same_first_name:
                        redirect_map[row['id']] = cand_id
                        redirect_targets['Person'] += 1
                        context_matches += 1
                        matched = True
                        break
                if matched:
                    break

    body_relations = [r for r in relation_rows if r.get('source') != 'header']

    redirected_relations = []
    for r in body_relations:
        r2 = dict(r)
        r2['start_id'] = redirect_map.get(r['start_id'], r['start_id'])
        r2['end_id'] = redirect_map.get(r['end_id'], r['end_id'])
        redirected_relations.append(r2)
    redirected_relations = [r for r in redirected_relations if r['start_id'] != r['end_id']]

    by_key = {}
    order = []
    for r in redirected_relations:
        key = (r['start_id'], r['end_id'], r['relation'], r.get('context', ''))
        if key not in by_key:
            by_key[key] = dict(r)
            order.append(key)
        else:
            # Redirect made this relation an exact duplicate of one already
            # kept -- don't just drop its 'source' mailNum, union it onto
            # the surviving row instead.
            existing_sources = [s.strip() for s in str(by_key[key].get('source', '')).split(';') if s.strip()]
            new_sources = [s.strip() for s in str(r.get('source', '')).split(';') if s.strip()]
            for s in new_sources:
                if s not in existing_sources:
                    existing_sources.append(s)
            by_key[key]['source'] = ';'.join(existing_sources)
    redirected_relations = [by_key[k] for k in order]

    referenced_ids = {r['start_id'] for r in redirected_relations} | {r['end_id'] for r in redirected_relations}

    body_entities = []
    for row in entity_rows:
        if row['type'] in HEADER_ONLY_ENTITY_TYPES:
            continue
        if row['id'] in redirect_map:
            continue
        if row['type'] == 'Person' and row['id'] not in referenced_ids:
            continue
        body_entities.append(row)

    org_name_to_id = {}
    affiliation_relations = []
    final_entities = []
    retype_id_map = {}
    matched_existing_header_org = 0
    retyped_to_team = 0
    salutations_stripped = 0
    for entity in body_entities:
        if entity['type'] != 'Person':
            final_entities.append(entity)
            continue
        try:
            props = json.loads(entity['properties'])
        except Exception:
            props = {}

        entity = dict(entity)
        if looks_like_team_name(props.get('personName')):
            new_id = f"tm{len(retype_id_map) + 1}"
            retype_id_map[entity['id']] = new_id
            entity['id'] = new_id
            entity['type'] = 'Team'
            if 'personName' in props:
                props['teamName'] = props.pop('personName')
            retyped_to_team += 1
        elif props.get('personName'):
            stripped = strip_salutation_if_full_name(props['personName'])
            if stripped != props['personName']:
                props['personName'] = stripped
                salutations_stripped += 1

        affiliation = props.pop('affiliation', None)
        if not affiliation:
            entity['properties'] = json.dumps(props)
            final_entities.append(entity)
            continue

        entity['properties'] = json.dumps(props)
        final_entities.append(entity)

        segments = [s.strip() for s in affiliation.split(',') if s.strip()]
        if not segments:
            continue
        org_name = segments[-1]
        role = segments[0] if len(segments) > 1 else None
        key = org_name.lower()
        if key in header_org_by_name:
            org_id = header_org_by_name[key]
            matched_existing_header_org += 1
        else:
            if key not in org_name_to_id:
                org_name_to_id[key] = f"og{len(org_name_to_id) + 1}"
                final_entities.append({
                    "id": org_name_to_id[key], "type": "Organization",
                    "properties": json.dumps({"orgName": org_name}),
                })
            org_id = org_name_to_id[key]
        affiliation_relations.append({
            "start_id": entity['id'], "end_id": org_id, "relation": "affiliation",
            "context": f"{role} at {org_name}" if role else org_name, "source": "",
        })
    body_entities = final_entities
    redirected_relations = [
        {**r, "start_id": retype_id_map.get(r['start_id'], r['start_id']),
         "end_id": retype_id_map.get(r['end_id'], r['end_id'])}
        for r in redirected_relations
    ]
    redirected_relations = redirected_relations + affiliation_relations

    # 'source' was an opaque Email entity id (e.g. "e517") -- replace it with
    # the actual mailNum, which is directly meaningful and matches the
    # provenance convention header_signature_parser.py already uses
    # (source_mailNums). Relations with no resolvable Email source (e.g. the
    # affiliation relations synthesized above, which aren't tied to one
    # specific email) keep whatever they already had.
    redirected_relations = [
        {**r, "source": email_id_to_mailnum.get(r['source'], r['source'])}
        for r in redirected_relations
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    ent_out = os.path.join(args.output_dir, "body_entities.csv")
    rel_out = os.path.join(args.output_dir, "body_relations.csv")

    with open(ent_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=entity_rows[0].keys(), quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(body_entities)

    with open(rel_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=relation_rows[0].keys(), quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(redirected_relations)

    dropped_entities = len(entity_rows) - len(body_entities)
    dropped_relations = len(relation_rows) - len(redirected_relations)
    print(f"Entities : {len(body_entities)} / {len(entity_rows)} kept "
          f"({dropped_entities} dropped) -> {ent_out}")
    print(f"Relations: {len(redirected_relations)} / {len(relation_rows)} kept "
          f"({dropped_relations} dropped) -> {rel_out}")
    print(f"Persons redirected onto header owner: {len(redirect_map)} "
          f"(-> Person: {redirect_targets['Person']}, -> Team: {redirect_targets['Team']}; "
          f"{name_affil_matches} by name+affiliation, {context_matches} by same-email-context name match)")
    print(f"Person.affiliation properties converted to affiliation relations: "
          f"{len(affiliation_relations)} (-> {len(org_name_to_id)} new Organization entities, "
          f"{matched_existing_header_org} reused an existing header Organization)")
    print(f"Persons retyped to Team (name looked institutional, not personal): {retyped_to_team}")
    print(f"Person salutations stripped (full name only, e.g. 'Dr. X Y' -> 'X Y'): {salutations_stripped}")


if __name__ == "__main__":
    main()
