# -*- coding: utf-8 -*-
"""
Header/signature-only regex parser for the revised EmailID/Team/Organization
ontology. Extracts entities and relations ONLY from email headers (Thread ID,
Mail ID, Date, From, To, Cc) and the closing signature block of the body
(e.g. "Best regards, ... "). Never reads the rest of the body text.

Entities  : Email(mailNum, mailDate), MailThread(threadID), EmailID(eID),
            Person(personName), Team(teamName), Organization(orgName, orgLoc)
Relations : (Email)-[:partOf]->(MailThread)
            (EmailID)-[:hasOwner]->(Person|Team)
            (Person)-[:memberOf {role,date}]->(Team)
            (Person)-[:affiliation {role,date}]->(Organization|Journal|Conference)
            (Team)-[:affiliation {date}]->(Organization|Journal|Conference)
            (Email)-[:sentBy]->(EmailID)
            (Email)-[:receivedBy]->(EmailID)
All dates are normalized to ISO (YYYY-MM-DD) via parse_date().

Person vs. Team classification, and every other heuristic here, is documented
inline next to the regex that implements it -- see classify_participant().

Two-pass name resolution (see collect_best_names): before any entity is
created, the WHOLE corpus is scanned once to find the best name ever
available for each address -- a real header display name (from ANY email),
else a signature-derived name (from whichever email had it as sender), else
the address itself. This stops an address first seen as a bare To/Cc
recipient (no name, no signature to draw from) from permanently locking in
the raw address as its Team/Person name, even though a later email where the
same address is the sender reveals a real name.

Optionally cross-references an already-produced entities_final.csv (the body
pipeline's output) to find which Journal/Conference was extracted from THIS
SAME email's body (via the nearest preceding Email row in file order -- see
load_venue_cross_reference), and adds a Person -[:affiliation]-> Journal/
Conference relation only when the SIGNATURE names an actual individual
different from the header's own From sender (typically a Team/shared-inbox
sender individually signed by a named person) -- never a blanket "sender
mentioned this venue somewhere in the body" link, and never for To/Cc
recipients.

Usage:
    python header_signature_parser.py \
        --input_file    data/PATRA/PATRA.txt \
        --entities_csv  <run>/final_outputs/entities_final.csv \
        --output_dir    data/header_extractions/
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# HEADER FIELD REGEXES (mirrors kg_extraction_pipeline.py's extract_header_info)
# ==============================================================================

# All four patterns below are anchored with ^ (used with re.MULTILINE in
# extract_header_info) so the keyword must be the FIRST thing on its own
# line. Without that anchor, re.search matches the keyword's first
# occurrence ANYWHERE in the text -- and a real Gmail-exported email can
# carry a "reply-To:" header line, whose "To:" substring is not at the
# start of its line but still satisfies an unanchored r'To:\s*...' match.
# Confirmed on MyPATRA.txt: a "reply-To: dmkd@wiley.com" line between From
# and the real To: caused the 'from' capture to swallow past it (its own
# lookahead only breaks on "\nTo:", which "\nreply-To:" doesn't match) and
# then the 'to' pattern's unanchored open match locked onto "To:" INSIDE
# "reply-To:" instead of the real recipient line below it -- cascading into
# garbled to/cc/date values for that entire email.
HEADER_FIELD_PATTERNS = [
    ('thread_id', r'^Thread ID:\s*(.+)'),
    ('mail_id',   r'^Mail ID:\s*(.+)'),
]

# Generic stop-boundary for the from/to/cc captures below: any line that
# LOOKS like a header label ("Word:" or "Multi Word:" at line start), not a
# hardcoded enumeration of specific field names. A real Gmail-exported email
# can interleave fields kg_extraction never anticipated (e.g. "reply-To:"
# between From and To, "Date:" between CC and Subject -- both confirmed in
# MyPATRA.txt) in any position, so this must stop at WHICHEVER label-shaped
# line comes next rather than assume a fixed From->To->Cc->Subject order or
# require every possible field name to be named up front.
_HEADER_LABELS = r'(?:[A-Za-z][A-Za-z \-]*:)'

FROM_TO_CC_PATTERNS = [
    ('from', r'^From:\s*(.+?)(?=\n' + _HEADER_LABELS + r'|\Z)'),
    ('to',   r'^To:\s*(.+?)(?=\n' + _HEADER_LABELS + r'|\Z)'),
    ('cc',   r'^Cc:\s*(.+?)(?=\n' + _HEADER_LABELS + r'|\Z)'),
]


MONTH_NAMES = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
    # 3-letter abbreviations -- a real Gmail-export corpus (e.g. MyPATRA.txt)
    # uses "29 Oct 2025, 23:07", unlike synthetic PATRA's spelled-out
    # "27th March 2025". "sept" kept as an extra alias for "sep".
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}


def parse_date(date_str: str) -> str:
    """Normalize a header Date value to ISO (YYYY-MM-DD). PATRA's actual
    format is ordinal day + month name + year (e.g. "27th March 2025",
    "3rd August 2019"); a real Gmail-export corpus instead uses abbreviated
    month + a trailing time (e.g. "29 Oct 2025, 23:07") -- both are handled,
    along with the digit-only formats, for robustness across corpora. Only
    the leading day/month/year is matched (no end-of-string anchor), so a
    trailing ", HH:MM" or similar is simply ignored rather than causing the
    whole value to fall through unparsed."""
    date_str = date_str.strip()
    match = re.match(r'(\d{1,2})-(\d{1,2})-(\d{4})$', date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})$', date_str)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    match = re.match(r'(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})\b', date_str, re.IGNORECASE)
    if match:
        day, month_name, year = match.groups()
        month_num = MONTH_NAMES.get(month_name.lower())
        if month_num:
            return f"{year}-{month_num:02d}-{int(day):02d}"
    return date_str


def extract_header_info(email_text: str) -> Dict[str, str]:
    header_info = {}
    for field, pattern in HEADER_FIELD_PATTERNS:
        match = re.search(pattern, email_text, re.IGNORECASE | re.MULTILINE)
        if match:
            header_info[field] = match.group(1).strip()

    match = re.search(r'^Date:\s*(.+)', email_text, re.IGNORECASE | re.MULTILINE)
    if match:
        header_info['date'] = parse_date(match.group(1).strip())

    for field, pattern in FROM_TO_CC_PATTERNS:
        match = re.search(pattern, email_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if match:
            header_info[field] = match.group(1).strip()

    match = re.search(r'^Subject:\s*(.+)', email_text, re.IGNORECASE | re.MULTILINE)
    if match:
        header_info['subject'] = match.group(1).strip()

    return header_info


def extract_body(email_text: str) -> str:
    """Same header-boundary detection as kg_extraction_pipeline.py."""
    header_pattern = re.compile(
        r'^(?:From|To|Cc|Bcc|Date|Subject|Thread-?ID|Mail-?ID|Message-?ID):',
        re.IGNORECASE
    )
    lines = email_text.splitlines()
    body_start_index = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if header_pattern.match(stripped):
            body_start_index = i + 1
        elif line.startswith((' ', '\t')) and i > 0 and body_start_index == i:
            body_start_index = i + 1
    body = "\n".join(lines[body_start_index:]).strip()
    if not body:
        match = re.search(r'\n\s*\n(.+)', email_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return body


# ==============================================================================
# PARTICIPANT (name, email) SPLITTING -- same three bracket styles observed in
# PATRA: "Name <email>", "Name [email]", "Name (email)", or a bare email.
# ==============================================================================

NAME_EMAIL_ANGLE_OR_BRACKET = re.compile(r'(.+?)\s*[<\[]([^>\]]+)[>\]]')
NAME_EMAIL_PAREN = re.compile(r'(.+?)\s*\(([^)]+)\)')


def split_participants(field_text: str) -> List[Tuple[Optional[str], Optional[str]]]:
    """Split a From/To/Cc header value into a list of (name, email) pairs."""
    out = []
    for part in re.split(r'[,;]', field_text.strip()):
        part = part.strip()
        if not part:
            continue
        name = email = None
        m = NAME_EMAIL_ANGLE_OR_BRACKET.search(part)
        if m:
            name, email = m.group(1).strip(), m.group(2).strip()
        else:
            m = NAME_EMAIL_PAREN.search(part)
            if m and '@' in m.group(2):
                name, email = m.group(1).strip(), m.group(2).strip()
            elif '@' in part:
                email = part
            else:
                name = part
        out.append((name, email))
    return out


# ==============================================================================
# SIGNATURE BLOCK EXTRACTION
# ==============================================================================

# Matches a whole line that ENDS in a closing keyword (regards/sincerely/
# thanks/cheers/best), with up to 4 arbitrary leading words -- covers the
# real variety seen in PATRA ("Best regards,", "With best regards,",
# "Warmest regards,", "With gratitude and best regards,", "Sincerely,",
# "With thanks and warm regards,", etc.) rather than an exact phrase list,
# which missed anything with a "With ..." prefix. Leading tokens allow "&"
# too (not just [A-Za-z]+) -- a real Gmail-exported sign-off commonly reads
# "Thanks & Regards," (confirmed as one real sender's standard closing
# across nearly every email in MyPATRA.txt), which the letters-only token
# pattern rejected outright, silently skipping signature/affiliation
# parsing for every email that used it.
CLOSING_RE = re.compile(
    r'^[ \t]*(?:[A-Za-z&]+[ \t]+){0,4}(?:regards|sincerely|thanks|thank you|cheers|best)[,.]?[ \t]*$',
    re.IGNORECASE | re.MULTILINE
)


def extract_signature_block(body: str) -> str:
    """
    Text after the LAST closing phrase (e.g. "Best regards,") to the end of
    the body. Used ONLY to derive the sender's role and institutional
    affiliation (see parse_signature_details() below) -- it does NOT
    override the header From/To/Cc display name used for owner identity/
    classification. Returns "" if no closing phrase is found.
    """
    matches = list(CLOSING_RE.finditer(body))
    if not matches:
        return ""
    return body[matches[-1].end():].strip()


EMAIL_RE = re.compile(r'[\w.+-]+@[\w.-]+\.\w+')

# A parenthetical aside or "On behalf of ..." line -- observed in PATRA
# signatures (e.g. "(on behalf of Prof. Ramesh Bhatia, ...)",
# "On behalf of all co-authors") sitting between the signer's name and their
# actual role/affiliation lines. Skipped when scanning for role/org text.
ASIDE_LINE_RE = re.compile(r'^\(.*\)$|^on behalf of\b', re.IGNORECASE)


def parse_signature_details(sig_block: str) -> Tuple[Optional[str], List[str]]:
    """
    Best-effort parse of the SENDER's own multi-line professional signature
    into (role, institutional_chain), grounded in the shapes actually seen
    in PATRA, e.g.:

        Ramesh Bhatia
        Professor, Department of Computer Science
        Indian Association for the Cultivation of Science
    ->  role="Professor", chain=["Department of Computer Science",
                                  "Indian Association for the Cultivation of Science"]

        Michael Bradley
        Assistant Professor, Stanford University
    ->  role="Assistant Professor", chain=["Stanford University"]

        Sunita Sen
        PhD Scholar
        Indian Association for the Cultivation of Science
    ->  role="PhD Scholar", chain=["Indian Association for the Cultivation of Science"]

    Line 0 (the signer's own name) is never used to rename/reclassify the
    header-resolved owner -- only role/chain are extracted here. Returns
    (None, []) when the block doesn't have this shape at all -- e.g. a bare
    first-name sign-off ("Best regards, Sunita") or a Team-only closing
    ("ACM JOCCH Editorial Office / jocch@acm.org").

    Some PATRA emails have a redundant informal-then-formal double
    sign-off, e.g. "Best regards,\n\nSunita\n\nSunita Sen\nPhD Scholar\n
    Department of Computer Science\nIndian Association for the Cultivation
    of Science" -- an extra repeated name line before the real role/chain
    content. Comma-free lines that plausibly repeat the signer's own name
    (see _looks_like_same_person) are skipped too, so the role/chain isn't
    misparsed as if "PhD Scholar" were the institution and the repeated
    "Sunita Sen" line were a role.
    """
    if not sig_block:
        return None, []
    lines = [l.strip() for l in sig_block.splitlines() if l.strip()]
    if len(lines) < 2:
        return None, []

    idx = 1  # always skip line 0 = signer's own name
    while idx < len(lines) and ',' not in lines[idx] and _looks_like_same_person(lines[0], lines[idx]):
        idx += 1

    content_lines = []
    for l in lines[idx:]:
        if ASIDE_LINE_RE.match(l) or EMAIL_RE.search(l) or l.lower().startswith('http'):
            continue
        content_lines.append(l)
    if not content_lines:
        return None, []

    segments = [s.strip() for s in content_lines[0].split(',')]
    segments.extend(content_lines[1:])
    if len(segments) < 2:
        return None, []  # no comma on line 1 AND no further line -> can't split role from org

    # If the FIRST segment itself reads as a department/team (e.g. Ramesh's
    # signature sometimes omits his role entirely and goes straight to
    # "Department of Computer Science" / "Indian Association for the
    # Cultivation of Science"), it isn't a role at all -- treat the whole
    # segment list as the institutional chain, with role left unstated,
    # rather than misappropriating it as text.
    if TEAM_KEYWORD_RE.search(segments[0]):
        return None, [s for s in segments if s]

    role, chain = segments[0], [s for s in segments[1:] if s]
    return role, chain


def _looks_like_same_person(name_line: str, candidate: str) -> bool:
    """
    True if `candidate` plausibly repeats/abbreviates the same person as
    `name_line` -- e.g. "Sunita" vs "Sunita Sen", or "Dr. Ananya Chatterjee"
    (header) vs "Ananya Chatterjee" (her own signature, honorific dropped).
    Honorifics are stripped from both sides before comparing so a title-only
    difference is never mistaken for two different people. Used both to
    skip a redundant repeated name line in a signature, and to gate the
    Journal/Conference affiliation rule (see resolve_signature_venue_affiliation)
    -- never to rename anything.
    """
    a = PERSON_TITLE_RE.sub('', name_line).lower().strip()
    b = PERSON_TITLE_RE.sub('', candidate).lower().strip()
    if not a or not b:
        return False
    return a == b or a in b.split() or b in a.split() or b.startswith(a) or a.startswith(b)


def classify_affiliation_segment(segment: str) -> Tuple[str, str, Optional[str]]:
    """
    Split one institutional-chain segment into (kind, name, loc):
      kind = 'Team' if it reads as a department/division/etc. (same
             TEAM_KEYWORD_RE used for header participants), else
             'Organization'.
      loc  = anything after a first comma within the segment itself (e.g.
             "Indian Association for the Cultivation of Science, Kolkata,
             India" -> loc="Kolkata, India"). Not observed in the sampled
             PATRA corpus (no signature line includes a location), so this
             will typically be None here -- kept for corpora that do.
    """
    parts = [p.strip() for p in segment.split(',')]
    name = parts[0]
    loc = ', '.join(parts[1:]) if len(parts) > 1 else None
    kind = "Team" if TEAM_KEYWORD_RE.search(name) else "Organization"
    return kind, name, loc


# ==============================================================================
# PERSON vs. TEAM CLASSIFICATION
# ==============================================================================

PERSON_TITLE_RE = re.compile(r'^(Dr|Prof|Professor|Mr|Mrs|Ms|Miss)\.?\s+', re.IGNORECASE)


def strip_honorific(name: str) -> str:
    """
    A salutation (Dr./Prof./Mr./Ms./...) is used ONLY as a classification
    signal (see classify_participant) -- never stored in personName. Storing
    it would make the same real person look like two different names across
    mentions (e.g. header "Dr. Ananya Chatterjee" vs her own signature
    "Ananya Chatterjee"), which is exactly the kind of divergence
    _looks_like_same_person otherwise has to paper over.
    """
    return PERSON_TITLE_RE.sub('', name).strip()

TEAM_KEYWORD_RE = re.compile(
    r'\b(committee|editorial|office|board|chairs?|organi[sz]ing|organizers?|'
    r'team|group|secretariat|desk|department|division|school|faculty|college|'
    r'submissions?|reviewers?|panel|council|tutorials?)\b',
    re.IGNORECASE
)
# NOTE: deliberately NOT "society"/"association" -- real institution proper
# names legitimately contain them (e.g. "Indian Association for the
# Cultivation of Science"), which caused a false Team classification when
# that name appeared as chain[1] (should always be Organization there).
#
# "school"/"faculty"/"college" added after auditing MyPATRA.txt: a real
# academic signature chain like "School of Cybersecurity" / "Old Dominion
# University" is a sub-unit + parent-institution pair exactly like
# "Department of Computer Science" / "Stanford University" -- but without
# this keyword, classify_affiliation_segment misread the sub-unit itself as
# the top-level Organization and never looked at the actual university at
# all. Confirmed real impact: Old Dominion University was completely absent
# from the graph, and so was Indian Association for the Cultivation of
# Science (the primary correspondent's own institution) via the same gap in
# parse_signature_details' role-vs-chain split.

ROLE_LOCALPART_RE = re.compile(
    r'^(submissions?|editorial|info|admin|no-?reply|contact|office|chairs?|'
    r'committee|support|help-?desk|secretariat)\b',
    re.IGNORECASE
)

BARE_ACRONYM_YEAR_RE = re.compile(r'^[A-Z]{2,8}(?:\s?\d{2,4})?$')

# Lowercase name particles common in real (non-synthetic) surnames -- "van
# Erp", "von Neumann", "de la Cruz", etc. classify_participant's own
# token-shape check otherwise requires EVERY token to start uppercase,
# which synthetic PATRA's name roster never had reason to violate but a
# real corpus does (confirmed: "Marieke van Erp" was misclassified "Team"
# solely because of "van", which then blocked her being recognized as one
# of two co-signers on an ISWC notification email -- see split_co_signers).
NAME_PARTICLES = {"van", "von", "der", "den", "de", "la", "le", "di", "da", "del", "dos", "das", "du"}

# A DELIBERATELY NARROWER subset of TEAM_KEYWORD_RE's words, used only to
# reject a candidate signature ROLE line that's actually an office/team
# name (see apply_venue_affiliation) -- NOT the full TEAM_KEYWORD_RE, which
# also includes "chairs?"/"reviewers?"/"panel"/"tutorials?"/"submissions?".
# Those ARE legitimate, common personal academic titles ("Track Chair",
# "Program Chair", "Reviewer") in a way "editorial office"/"secretariat"
# never are -- reusing the full list here rejected "ISWC2026 Resource
# Track Chairs" as a candidate role purely because of "Chairs", even though
# it's genuinely the two signers' own title, not a team/office name.
OFFICE_NOT_ROLE_RE = re.compile(
    r'\b(committee|editorial|office|board|organi[sz]ing|organizers?|'
    r'team|group|secretariat|desk|department|division|school|faculty|college)\b',
    re.IGNORECASE
)


def classify_participant(name: str, email: Optional[str]) -> str:
    """
    Return 'Person' or 'Team' for a header/signature display name.
    See the module docstring / conversation notes for the full rationale;
    summary:
      1. Honorific title (Dr./Prof./Mr./Ms./...)      -> Person (overrides all)
      2. Institutional keyword in the name             -> Team
      3. Role-shaped local-part (submissions@, info@)  -> Team
      4. Bare "ACRONYM YEAR" with nothing else          -> Team (ambiguous;
         usually a Conference/Journal's own display name reused as sender;
         classified/named from the header text as-is, no override).
      5. Default: short Title-Case name, no digits      -> Person
    """
    name = (name or "").strip()
    local_part = email.split('@')[0] if email else ""

    if PERSON_TITLE_RE.match(name):
        return "Person"
    if TEAM_KEYWORD_RE.search(name):
        return "Team"
    if ROLE_LOCALPART_RE.match(local_part):
        return "Team"
    if BARE_ACRONYM_YEAR_RE.match(name):
        return "Team"

    tokens = name.split()
    if tokens and 1 <= len(tokens) <= 4 and all(
        re.match(r"^[A-Z][\w.'-]*$", t) or t.lower() in NAME_PARTICLES for t in tokens
    ):
        return "Person"
    return "Team"


def split_co_signers(name: str) -> List[str]:
    """
    Split a signature name line that may name MULTIPLE people (e.g.
    "Marieke van Erp and Axel Polleres", "X, Y and Z") into individual
    names. Used as a fallback ONLY when the whole line fails
    classify_participant's Person check -- a real co-signed line like this
    one has 6 tokens total, over the <=4-token cap that check uses (by
    design, to avoid classifying a long institution name as a Person), so
    it gets called "Team" as a whole even though it's naming two actual
    people. Splitting first and re-classifying each piece independently
    (see apply_venue_affiliation) is what tells a genuine multi-signer line
    apart from an actual Team/office name that happens to contain "and".
    """
    parts = re.split(r'\s*,\s*|\s+and\s+', name)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [name]


# ==============================================================================
# ORGANIZATION-PREFIX SPLITTING (explicit allowlist, not a domain guess)
# ==============================================================================

KNOWN_ORG_PREFIXES = re.compile(
    r'^(ACM|IEEE|ACL|USENIX|NSF|Springer|Elsevier|SIGIR|SIGCHI)\b\s*(.*)$'
)


def split_org_prefix(team_name: str) -> Tuple[Optional[str], str]:
    """
    If a Team display name is led by a known parent-organization acronym
    (e.g. "ACM JOCCH Editorial Office"), split into
    (orgName="ACM", remaining team name="JOCCH Editorial Office").
    Returns (None, team_name) if no known prefix matches -- deliberately a
    small explicit allowlist rather than a guess (e.g. from an email domain),
    per "only extract information that is actually available in the text."
    """
    m = KNOWN_ORG_PREFIXES.match(team_name.strip())
    if m and m.group(2):
        return m.group(1), m.group(2).strip()
    return None, team_name


# ==============================================================================
# ENTITY / RELATION REGISTRIES
# ==============================================================================

PREFIX_MAP = {"EmailID": "eid", "Person": "pn", "Team": "tm", "Organization": "og",
              "Email": "e", "MailThread": "t"}

entities: Dict[str, Dict] = {}       # id -> {"type", "properties", "source_mailNums": set}
id_counters: Dict[str, int] = defaultdict(int)
address_to_owner_id: Dict[str, str] = {}   # lowercased email address -> Person/Team id
name_to_org_id: Dict[str, str] = {}        # lowercased orgName -> Organization id
name_to_team_id: Dict[str, str] = {}       # lowercased teamName -> Team id (no address of its
                                            # own; used for memberOf-only Teams found in a
                                            # signature's institutional chain, e.g. "Department
                                            # of Computer Science" -- distinct from Teams that
                                            # own an EmailID)
email_id_registry: Dict[str, str] = {}     # lowercased email address -> EmailID id
own_email_registry: Dict[str, str] = {}    # mailNum -> this script's own Email entity id
thread_registry: Dict[str, str] = {}       # threadID -> MailThread entity id
name_to_signature_person_id: Dict[str, str] = {}  # lowercased name -> Person id, for a named
                                            # individual found ONLY in a Team-sender's
                                            # signature (no address of their own -- see
                                            # resolve_signature_venue_affiliation)
relations: List[Dict] = []           # {"start","end","relation","properties","source_mailNums"}

# Two-pass name resolution (see collect_best_names): lowercased address ->
# the best display name ever found for it across the WHOLE corpus, before
# any entity is created. Fixes the case where an address is FIRST seen as a
# bare To/Cc recipient (no name, no signature available) but is later a
# sender with a real header name or signature -- without this, first-seen-
# wins dedup would permanently lock in the address itself as the name.
header_names: Dict[str, str] = {}
signature_names: Dict[str, str] = {}


def collect_best_names(all_emails: List[str]) -> None:
    """
    Pass 1 over the whole corpus (before any entity is created): populate
    `header_names`/`signature_names` with the best name ever available for
    each address. Priority used later at entity-creation time is
    header_names > signature_names > the raw address itself -- a header
    display name is never overridden by a signature (see resolve_owner），
    but when NO email ever gives this address a header name, a signature
    from whichever email had it as sender (and a usable, non-address first
    signature line) is used instead of falling back to the bare address.
    """
    for email_text in all_emails:
        header = extract_header_info(email_text)

        for field in ('from', 'to', 'cc'):
            if field not in header:
                continue
            for name, email in split_participants(header[field]):
                if not email or not name:
                    continue
                addr = email.lower().strip()
                stripped = name.strip()
                if stripped and not EMAIL_RE.fullmatch(stripped) and addr not in header_names:
                    header_names[addr] = stripped

        if 'from' not in header:
            continue
        from_pairs = split_participants(header['from'])
        if not from_pairs or not from_pairs[0][1]:
            continue
        addr = from_pairs[0][1].lower().strip()
        if addr in signature_names:
            continue
        body = extract_body(email_text)
        sig_block = extract_signature_block(body)
        if not sig_block:
            continue
        sig_lines = [l.strip() for l in sig_block.splitlines() if l.strip()]
        if sig_lines and not EMAIL_RE.fullmatch(sig_lines[0]) and not sig_lines[0].lower().startswith('http'):
            signature_names[addr] = sig_lines[0]


def best_name_for_address(addr: str) -> str:
    return header_names.get(addr) or signature_names.get(addr) or addr


def _mint_id(entity_type: str) -> str:
    id_counters[entity_type] += 1
    return f"{PREFIX_MAP[entity_type]}{id_counters[entity_type]}"


def get_or_create_entity(entity_type: str, properties: dict, key: str,
                          registry: Dict[str, str], mail_num: str) -> str:
    """
    `registry` maps `key` -> entity id for this entity_type's own identity
    anchor (email address for Person/Team via EmailID ownership; orgName for
    Organization; the eID string itself for EmailID). First occurrence of a
    given key wins the canonical properties; later occurrences just extend
    source_mailNums (see conversation note on first-seen-wins dedup).
    """
    if key in registry:
        eid = registry[key]
        entities[eid]["source_mailNums"].add(mail_num)
        return eid
    eid = _mint_id(entity_type)
    entities[eid] = {"type": entity_type, "properties": properties,
                      "source_mailNums": {mail_num}}
    registry[key] = eid
    return eid


# The identifying property used as a human-readable label per entity type,
# for the start_label/end_label columns in the output relations file.
LABEL_PROPERTY = {
    "Email": "mailNum", "EmailID": "eID", "Person": "personName",
    "Team": "teamName", "Organization": "orgName", "MailThread": "threadID",
    "Journal": "journalTitle", "Conference": "confTitle",
}


def _default_label(entity_id: str, entity_type: str) -> str:
    """Look up entity_id's own label property from this script's `entities`
    registry -- only works for entity_type's this script itself creates."""
    prop = LABEL_PROPERTY.get(entity_type)
    if not prop:
        return ""
    return entities.get(entity_id, {}).get("properties", {}).get(prop, "")


def add_relation(start_id: str, end_id: str, relation: str, mail_num: str,
                  properties: Optional[dict] = None, dedup: bool = True,
                  start_type: Optional[str] = None, end_type: Optional[str] = None,
                  start_label: Optional[str] = None, end_label: Optional[str] = None):
    """
    `start_type`/`end_type` and `start_label`/`end_label` describe the
    endpoints in the output relations file. When omitted, they're looked up
    from this script's own `entities` registry -- pass them explicitly for
    an endpoint that lives in an EXTERNAL file instead (the cross-referenced
    entities_final.csv's Journal/Conference ids, which never get an entry in
    `entities` here).
    """
    properties = {k: v for k, v in (properties or {}).items() if v}
    start_type = start_type or entities.get(start_id, {}).get("type", "UNKNOWN")
    end_type = end_type or entities.get(end_id, {}).get("type", "UNKNOWN")
    if start_label is None:
        start_label = _default_label(start_id, start_type)
    if end_label is None:
        end_label = _default_label(end_id, end_type)
    if dedup:
        for r in relations:
            if r["start"] == start_id and r["end"] == end_id and r["relation"] == relation:
                r["source_mailNums"].add(mail_num)
                r["properties"].update(properties)
                return
    relations.append({"start": start_id, "start_type": start_type, "start_label": start_label,
                       "end": end_id, "end_type": end_type, "end_label": end_label,
                       "relation": relation,
                       "properties": properties, "source_mailNums": {mail_num}})


# ==============================================================================
# JOURNAL/CONFERENCE CROSS-REFERENCE (optional)
# ==============================================================================

def load_venue_cross_reference(entities_csv: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    entities_final.csv is written in the body-pipeline's per-email
    processing order, with each email's own Email row (properties.mailNum)
    appearing among the entities it produced. So the nearest PRECEDING
    Email row for any given entity row is, in practice, that entity's own
    source email -- no need to cross-reference relations_final.csv's
    `source` column at all.

    Returns mailnum_to_venues : mailNum -> [(venue_id, venue_type,
    venue_label), ...] for every Journal/Conference row seen since that
    Email row -- venue_label is that row's own journalTitle/confTitle,
    carried along here since this Journal/Conference entity lives in the
    EXTERNAL file, not this script's own `entities` registry.

    NOTE: entities are globally deduped by the body pipeline (one row per
    entity, ever) -- a Journal/Conference only gets a row on its FIRST
    mention across the whole corpus. A later email that references the same
    venue without minting a new entity id won't be captured here.
    """
    mailnum_to_venues: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    current_mailnum = None
    with open(entities_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                props = json.loads(row['properties'])
            except (json.JSONDecodeError, TypeError):
                props = {}
            if row['type'] == 'Email' and 'mailNum' in props:
                current_mailnum = str(props['mailNum'])
            elif row['type'] in ('Journal', 'Conference') and current_mailnum:
                label = props.get(LABEL_PROPERTY.get(row['type']), "")
                mailnum_to_venues[current_mailnum].append((row['id'], row['type'], label))
    return mailnum_to_venues


# ==============================================================================
# PER-EMAIL PROCESSING
# ==============================================================================

def reset_state() -> None:
    """Clear all module-level registries so this module can be reused
    within a single process across multiple runs (e.g. by an orchestrator
    that imports it rather than shelling out to a fresh interpreter)."""
    entities.clear()
    id_counters.clear()
    address_to_owner_id.clear()
    name_to_org_id.clear()
    name_to_team_id.clear()
    email_id_registry.clear()
    own_email_registry.clear()
    thread_registry.clear()
    name_to_signature_person_id.clear()
    relations.clear()
    header_names.clear()
    signature_names.clear()


CONTEXT_ID_PREFIX = "HDR_"


def build_llm_context(participants: List[Tuple[str, str, dict]], org_email_hints: List[str],
                       mail_date: Optional[str] = None, subject: Optional[str] = None
                       ) -> Tuple[str, Dict[str, str]]:
    """
    Format the Persons/Teams this module has already resolved for ONE
    email's From/To/Cc into a prompt-context string for the body-extraction
    LLM, so it reuses these ids instead of minting its own redundant ones.
    `participants` is (entity_id, entity_type, properties) as collected by
    process_email_header(); `org_email_hints` are the addresses among them
    that resolved to a Team (candidate journalMail/confMail values -- see
    kg_extraction_pipeline.py's org-mail auto-attach).

    IDs are shown to the LLM with a "HDR_" prefix (e.g. "HDR_pn6"), not this
    module's raw id ("pn6") -- on a small corpus, this module's own ids stay
    numerically small (e.g. pn1..pn7 for a 15-email corpus), squarely inside
    the range the LLM independently invents for genuinely NEW entities it
    finds in THIS email's body (its own local temp-id counter also starts at
    1 each call, since it has no memory of other calls). Without a distinct
    namespace, a real collision is not just possible but likely at small
    scale: confirmed on MyPATRA.txt, where the LLM reused "pn7" (context-
    reserved for a specific header Person) as its own local temp id for an
    unrelated newly-found Person in the same response, silently corrupting
    every relation in that response meaning to reference the header Person.
    (On the ~1000-email PATRA corpus this never surfaces, since by the time
    header ids reach a given email they're already numbered in the hundreds,
    far past anything the LLM's own small per-call numbering would produce
    -- but that was luck of scale, not a real guarantee.)

    Returns (context_text, context_id_map) where context_id_map maps each
    displayed "HDR_xxx" string to the real id "xxx" -- pass straight through
    to kg_extraction_pipeline.process_email()'s `context_id_map` parameter.
    """
    persons = [(eid, props) for eid, etype, props in participants if etype == "Person"]
    teams = [(eid, props) for eid, etype, props in participants if etype == "Team"]
    context_id_map = {f"{CONTEXT_ID_PREFIX}{eid}": eid for eid, _etype, _props in participants}
    header_lines = ""
    if mail_date:
        header_lines += f"This email's date: {mail_date}\n"
    if subject:
        # The Subject line is stripped out of body_text before this LLM
        # call ever sees it, but it's often the ONLY place a SubmissionID
        # is stated (e.g. "Decision on Manuscript ID DMKD-00644.R1" never
        # repeats the ID anywhere in the body itself) -- confirmed on
        # MyPATRA.txt: without this, an accepted paper's PaperStatus had no
        # way to link back to its SubmissionID at all.
        header_lines += f"This email's subject line: {subject}\n"
    if not persons and not teams and not org_email_hints:
        return header_lines + "No persons in headers.", context_id_map
    context = header_lines
    if persons:
        context += ("Persons already extracted from email headers/signatures. If this email "
                    "involves one of them, reuse their id EXACTLY AS SHOWN below (including the "
                    "HDR_ prefix) in your own output -- do not invent a different id for them, "
                    "and do not reuse an id shown here for any OTHER, different entity:\n")
        for eid, props in persons:
            context += f"- {CONTEXT_ID_PREFIX}{eid}: {json.dumps(props)}\n"
    if teams:
        context += ("Teams/organizational senders or recipients already extracted from email "
                    "headers (reuse these ids EXACTLY AS SHOWN, HDR_ prefix included; do not "
                    "create a Person for these):\n")
        for eid, props in teams:
            context += f"- {CONTEXT_ID_PREFIX}{eid}: {json.dumps(props)}\n"
    if org_email_hints:
        context += ("Organizational sender/recipient addresses in this email's headers "
                    "(their Team entity is listed above by id):\n")
        for addr in org_email_hints:
            context += f"- {addr}\n"
    return context, context_id_map


def process_email_header(email_text: str, email_num: Optional[int] = None) -> Optional[dict]:
    """
    Header+signature entity/relation pass for ONE email, EXCLUDING the
    Journal/Conference venue-affiliation step (see apply_venue_affiliation)
    -- that step alone needs to know which venues the body-extraction LLM
    found in this same email, so it's deferred until after body extraction
    has run for the whole corpus. Everything else here (Email, MailThread,
    EmailID, Person, Team, Organization, hasOwner/sentBy/receivedBy/partOf/
    memberOf/affiliation-from-signature-chain) has no such dependency and
    can run before, or interleaved with, body extraction.

    Returns None if this email has no Mail ID (nothing to anchor entities
    to). Otherwise returns a dict of everything apply_venue_affiliation()
    needs later, plus `participants` (for build_llm_context) and
    `own_email_id` (the Email entity id body extraction should cite as its
    relations' `source`, replacing kg_extraction_pipeline.py's own).
    """
    header = extract_header_info(email_text)
    # Real corpora (e.g. a Gmail export) don't carry synthetic-PATRA's own
    # "Mail ID:"/"Thread ID:" header lines at all -- fall back to this
    # email's 1-indexed position in the corpus so entities still have a
    # stable per-email anchor. Never fires for PATRA.txt-shaped corpora
    # (they always have a real Mail ID), so existing behavior there is
    # unchanged.
    mail_num = header.get('mail_id') or (str(email_num) if email_num is not None else None)
    if not mail_num:
        return None
    mail_date = header.get('date')
    subject = header.get('subject')
    body = extract_body(email_text)
    sig_block = extract_signature_block(body)

    def resolve_owner(email: Optional[str]) -> Optional[str]:
        """
        Resolve an address to a Person/Team entity id. The display name
        used is whatever collect_best_names() (pass 1, over the WHOLE
        corpus) resolved for this address beforehand: a real header display
        name if ANY email ever gave one (never overridden by a signature --
        e.g. a From line reading "ACM Journal on Computing and Cultural
        Heritage <jocch@acm.org>" stays classified/named from that text
        alone); else a signature-derived name from whichever email had this
        address as sender; else the raw address itself. This is why an
        address first seen as a bare To/Cc recipient (no name available at
        all) doesn't permanently lock in the address as its name once a
        later email reveals a real one. Signature-derived role/affiliation/
        memberOf is handled separately in resolve_signature_affiliation(),
        for the sender only.
        """
        if not email:
            return None
        addr = email.lower().strip()
        if addr in address_to_owner_id:
            entities[address_to_owner_id[addr]]["source_mailNums"].add(mail_num)
        else:
            display_name = best_name_for_address(addr)
            ptype = classify_participant(display_name, email)
            if ptype == "Team":
                org_name, remaining = split_org_prefix(display_name)
                owner_id = get_or_create_entity(
                    "Team", {"teamName": remaining}, addr, address_to_owner_id, mail_num
                )
                if org_name:
                    org_id = get_or_create_entity(
                        "Organization", {"orgName": org_name}, org_name.lower(),
                        name_to_org_id, mail_num
                    )
                    add_relation(owner_id, org_id, "affiliation", mail_num,
                                 {"date": mail_date})
            else:
                owner_id = get_or_create_entity(
                    "Person", {"personName": strip_honorific(display_name)}, addr,
                    address_to_owner_id, mail_num
                )

        return address_to_owner_id[addr]

    def resolve_signature_affiliation(owner_id: str):
        """
        Person-only: parse the sender's own signature (role + institutional
        chain, see parse_signature_details()) and emit memberOf/affiliation
        relations for THIS EMAIL's sender. A signature only ever describes
        its own signer, so this is never applied to To/Cc recipients.
        """
        if entities[owner_id]["type"] != "Person":
            return
        role, chain = parse_signature_details(sig_block)
        if not chain:
            return
        kind, name, loc = classify_affiliation_segment(chain[0])
        if kind == "Team":
            # A generic department/team name (e.g. "Department of Computer
            # Science") is not globally unique -- Stanford, Oxford, and IACS
            # all have one. Scope the dedup key to (team name, parent org
            # name) so they don't collide into a single false-merged Team
            # affiliated with multiple unrelated Organizations.
            org_kind = org_name = org_loc = None
            if len(chain) >= 2:
                org_kind, org_name, org_loc = classify_affiliation_segment(chain[1])
            team_key = f"{name.lower().strip()}|{(org_name or '').lower().strip()}"
            team_id = get_or_create_entity(
                "Team", {"teamName": name}, team_key, name_to_team_id, mail_num
            )
            add_relation(owner_id, team_id, "memberOf", mail_num,
                         {"role": role, "date": mail_date})
            if org_name:
                org_props = {"orgName": org_name, **({"orgLoc": org_loc} if org_loc else {})}
                org_id = get_or_create_entity(
                    "Organization", org_props, org_name.lower().strip(),
                    name_to_org_id, mail_num
                )
                add_relation(team_id, org_id, "affiliation", mail_num, {"date": mail_date})
        else:
            org_props = {"orgName": name, **({"orgLoc": loc} if loc else {})}
            org_id = get_or_create_entity(
                "Organization", org_props, name.lower().strip(), name_to_org_id, mail_num
            )
            add_relation(owner_id, org_id, "affiliation", mail_num,
                         {"role": role, "date": mail_date})

    def get_or_create_email_id(address: str) -> str:
        key = address.lower().strip()
        return get_or_create_entity("EmailID", {"eID": key}, key, email_id_registry, mail_num)

    from_pairs = split_participants(header['from']) if 'from' in header else []
    to_pairs = split_participants(header['to']) if 'to' in header else []
    cc_pairs = split_participants(header['cc']) if 'cc' in header else []

    # This script's own Email entity (mailNum, mailDate) -- extracted here
    # even though the body pipeline's entities_final.csv already has one, so
    # header_entities.csv/header_relations.csv are self-contained and don't
    # depend on --entities_csv for sentBy/receivedBy to resolve.
    own_email_props = {"mailNum": mail_num, **({"mailDate": mail_date} if mail_date else {})}
    own_email_id = get_or_create_entity("Email", own_email_props, mail_num, own_email_registry, mail_num)

    thread_id = header.get('thread_id')
    if thread_id:
        thread_entity_id = get_or_create_entity(
            "MailThread", {"threadID": thread_id}, thread_id, thread_registry, mail_num
        )
        add_relation(own_email_id, thread_entity_id, "partOf", mail_num, dedup=True)

    participants: List[Tuple[str, str, dict]] = []
    org_email_hints: List[str] = []

    for _name, email in from_pairs:
        if not email:
            continue
        owner_id = resolve_owner(email)
        eid_id = get_or_create_email_id(email)
        add_relation(eid_id, owner_id, "hasOwner", mail_num, dedup=True)
        add_relation(own_email_id, eid_id, "sentBy", mail_num, dedup=False)
        resolve_signature_affiliation(owner_id)
        owner_type = entities[owner_id]["type"]
        participants.append((owner_id, owner_type, entities[owner_id]["properties"]))
        if owner_type == "Team":
            org_email_hints.append(email)

    for _name, email in to_pairs + cc_pairs:
        if not email:
            continue
        owner_id = resolve_owner(email)
        eid_id = get_or_create_email_id(email)
        add_relation(eid_id, owner_id, "hasOwner", mail_num, dedup=True)
        add_relation(own_email_id, eid_id, "receivedBy", mail_num, dedup=False)
        owner_type = entities[owner_id]["type"]
        participants.append((owner_id, owner_type, entities[owner_id]["properties"]))
        if owner_type == "Team":
            org_email_hints.append(email)

    return {
        "mail_num": mail_num, "mail_date": mail_date, "subject": subject, "sig_block": sig_block,
        "from_pairs": from_pairs, "own_email_id": own_email_id,
        "participants": participants, "org_email_hints": org_email_hints,
    }


def apply_venue_affiliation(pending: dict, venues: List[Tuple[str, str, str]]) -> None:
    """
    Journal/Conference affiliation -- NOT a blanket "sender mentioned this
    venue somewhere in the body" link (too permissive: nearly every email
    in a submission thread mentions the venue without implying
    affiliation). Only fires when the SIGNATURE names an actual individual
    who is a different person from the header's own From sender -- the
    common case being a Team/shared-inbox sender (e.g. "ACM JOCCH
    Editorial Office") individually signed by a named person underneath.

    Split out of process_email_header() because `venues` (what
    Journal/Conference the body-extraction LLM found in THIS email) is only
    known once body extraction has run -- call this in a second pass, after
    body extraction, over every pending dict process_email_header() returned.
    """
    mail_num = pending["mail_num"]
    mail_date = pending["mail_date"]
    sig_block = pending["sig_block"]
    from_pairs = pending["from_pairs"]
    if not (venues and sig_block and from_pairs):
        return
    header_from_name = (from_pairs[0][0] or '').strip()
    sig_lines = [l.strip() for l in sig_block.splitlines() if l.strip()]
    signer_line = sig_lines[0] if sig_lines else None
    if not (signer_line
            and not EMAIL_RE.fullmatch(signer_line)
            and not signer_line.lower().startswith('http')
            and not _looks_like_same_person(header_from_name, signer_line)):
        return

    if classify_participant(signer_line, None) == "Person":
        signer_names = [signer_line]
    else:
        # Might be several co-signers on one line (e.g. "Marieke van Erp
        # and Axel Polleres") rather than a genuine Team/office name --
        # only accept the split if EVERY resulting piece independently
        # classifies as a Person; otherwise this really is a Team and
        # splitting on "and"/"," would just produce garbage names.
        candidates = split_co_signers(signer_line)
        if len(candidates) > 1 and all(classify_participant(c, None) == "Person" for c in candidates):
            signer_names = candidates
        else:
            return

    # The signer's role/title, if stated, is the first "real" line after
    # their own name (e.g. "Editor-in-Chief, WIREs Data Mining and
    # Knowledge Discovery" -- take the segment before the first comma,
    # since anything after it just restates the venue we already have via
    # `venues`). Shared across all co-signers on the same line (e.g. "...
    # Resource Track Chairs" describes both people equally). `role` is a
    # declared property on `affiliation` in PERKOnto.json but was never
    # being set here at all -- confirmed real impact: a role-filtered query
    # for "Editor-in-Chief" over this relation always returned nothing,
    # even though the signature states the role in plain text right next
    # to the name.
    #
    # Rejected via OFFICE_NOT_ROLE_RE if the candidate line reads as an
    # office/team name rather than a personal title -- e.g. "WIREs Data
    # Mining and Knowledge Discovery Editorial Office" (no comma, so it
    # would otherwise be taken whole) is the sender's shared inbox/team,
    # not the individual signer's own role. Deliberately NOT the full
    # TEAM_KEYWORD_RE here -- see OFFICE_NOT_ROLE_RE's own comment for why
    # "chairs?"/"reviewers?"/etc. must NOT reject a real personal title
    # like "Resource Track Chairs".
    role = None
    for line in sig_lines[1:]:
        if ASIDE_LINE_RE.match(line) or EMAIL_RE.search(line) or line.lower().startswith('http'):
            continue
        candidate = line.split(',')[0].strip()
        if candidate and not OFFICE_NOT_ROLE_RE.search(candidate):
            role = candidate
        break

    for signer_name in signer_names:
        stripped_name = strip_honorific(signer_name)
        person_id = get_or_create_entity(
            "Person", {"personName": stripped_name}, stripped_name.lower(),
            name_to_signature_person_id, mail_num
        )
        for venue_id, venue_type, venue_label in venues:
            add_relation(person_id, venue_id, "affiliation", mail_num,
                         {"date": mail_date, "role": role},
                         end_type=venue_type, end_label=venue_label)


def process_email(email_text: str, mailnum_to_venues: Dict[str, List[Tuple[str, str, str]]],
                   email_num: Optional[int] = None) -> None:
    """Back-compat wrapper for standalone CLI use (see main()): runs the
    header pass and the venue-affiliation pass for one email back-to-back,
    against an already-fully-built `mailnum_to_venues` (the historical
    two-pass-via-CSV flow -- see load_venue_cross_reference)."""
    pending = process_email_header(email_text, email_num)
    if pending is None:
        return
    venues = mailnum_to_venues.get(pending["mail_num"], [])
    apply_venue_affiliation(pending, venues)


# ==============================================================================
# OUTPUT
# ==============================================================================

def _is_email_shaped_team(eid: str, data: dict) -> bool:
    """
    A Team whose teamName is itself an email address, e.g. teamName=
    "acl2021@softconf.com" -- happens when a header participant has NO
    display name at all (a bare address in From/To/Cc), so resolve_owner's
    `display_name = (name or addr).strip()` fallback used the raw address
    itself, and classify_participant's default branch called it a Team.
    Not a real team name -- dropped from the output entirely.
    """
    return data["type"] == "Team" and bool(EMAIL_RE.fullmatch(data["properties"].get("teamName", "")))


def write_outputs(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    dropped_ids = {eid for eid, data in entities.items() if _is_email_shaped_team(eid, data)}
    kept_entities = {eid: data for eid, data in entities.items() if eid not in dropped_ids}
    kept_relations = [r for r in relations if r["start"] not in dropped_ids and r["end"] not in dropped_ids]

    ent_path = os.path.join(output_dir, "header_entities.csv")
    with open(ent_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id", "type", "properties", "source_mailNums"])
        for eid, data in kept_entities.items():
            writer.writerow([
                eid, data["type"], json.dumps(data["properties"]),
                ";".join(sorted(data["source_mailNums"]))
            ])

    rel_path = os.path.join(output_dir, "header_relations.csv")
    with open(rel_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["start_id", "start_type", "start_label", "relation", "end_id",
                          "end_type", "end_label", "relProperties", "source_mailNums"])
        for r in kept_relations:
            writer.writerow([
                r["start"], r["start_type"], r["start_label"], r["relation"], r["end"],
                r["end_type"], r["end_label"],
                json.dumps(r["properties"]), ";".join(sorted(r["source_mailNums"]))
            ])

    print(f"Entities : {len(kept_entities)} -> {ent_path}"
          f" ({len(dropped_ids)} email-shaped Team name(s) dropped)")
    print(f"Relations: {len(kept_relations)} -> {rel_path}"
          f" ({len(relations) - len(kept_relations)} dropped along with them)")


def main():
    parser = argparse.ArgumentParser(description="Header/signature-only entity+relation parser")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--entities_csv", default=None,
                         help="Body-pipeline's entities_final.csv, for Journal/Conference cross-ref")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    mailnum_to_venues: Dict[str, List[Tuple[str, str, str]]] = {}
    if args.entities_csv:
        mailnum_to_venues = load_venue_cross_reference(args.entities_csv)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        all_emails = [e.strip() for e in f.read().split('EMAIL_END') if e.strip()]

    collect_best_names(all_emails)  # pass 1: resolve best names before any entity creation

    for i, email_text in enumerate(all_emails):
        process_email(email_text, mailnum_to_venues, email_num=i + 1)

    write_outputs(args.output_dir)


if __name__ == "__main__":
    main()
