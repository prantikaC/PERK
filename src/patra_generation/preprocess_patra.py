# -*- coding: utf-8 -*-
import re
import random
import string
import argparse
from datetime import datetime, timedelta

random.seed(42)


# ==============================================================================
# 1. STATISTICS CLASS
# ==============================================================================

class ProcessingStatistics:
    def __init__(self):
        self.total_emails = 0
        self.formatting_fixes = 0

        self.unique_normalized_subjects = set()
        self.new_threads_created = 0
        self.emails_grouped_into_existing_threads = 0
        self.new_unique_thread_ids = set()

        self.dates_found = 0
        self.dates_shifted = 0
        self.years_shifted = 0

        self.original_date_range = (None, None)
        self.new_date_range = (None, None)

    def set_ranges(self, orig_start, orig_end, new_start, new_end):
        self.original_date_range = (orig_start, orig_end)
        self.new_date_range = (new_start, new_end)

    def generate_report(self):
        print("\n" + "=" * 60)
        print("              CORRECTION STATISTICS REPORT")
        print("=" * 60)

        print(f"1. DATASET OVERVIEW")
        print(f"   - Total Emails Processed:            {self.total_emails}")
        print(f"   - Formatting Fixes (Newlines):       {self.formatting_fixes}")

        print(f"\n2. THREAD RECONSTRUCTION")
        print(f"   - Unique Normalized Subjects:        {len(self.unique_normalized_subjects)}")
        print(f"   - New Threads Created (Non-Reply):   {self.new_threads_created}")
        print(f"   - Replies Linked to Existing Threads:{self.emails_grouped_into_existing_threads}")
        print(f"   - Total Unique Thread IDs:           {len(self.new_unique_thread_ids)}")

        print(f"\n3. TEMPORAL SHIFTS")
        if self.original_date_range[0]:
            orig_s = self.original_date_range[0].strftime('%Y-%m-%d')
            orig_e = self.original_date_range[1].strftime('%Y-%m-%d')
            new_s = self.new_date_range[0].strftime('%Y-%m-%d')
            new_e = self.new_date_range[1].strftime('%Y-%m-%d')
            print(f"   - Original Range: {orig_s} to {orig_e}")
            print(f"   - Target Range:   {new_s} to {new_e}")

        print(f"\n4. CONTENT CORRECTIONS")
        print(f"   - Dates Shifted (Header & Body):     {self.dates_shifted}")
        print(f"   - Isolated Years Shifted:            {self.years_shifted}")
        print("=" * 60 + "\n")


# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def generate_thread_id(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def normalize_subject(subject):
    """Strip Re:/Fwd: prefixes and lowercase to find the canonical thread root."""
    if not subject:
        return ""
    clean = re.sub(r'^(Re:\s*|Fwd:\s*|Fw:\s*|Response to:\s*)+', '', subject, flags=re.IGNORECASE)
    return ' '.join(clean.split()).strip().lower()


def parse_date(date_str):
    date_str = date_str.strip()
    formats = [
        "%d-%m-%Y", "%d/%m/%Y",
        "%d %B %Y", "%d %b %Y",
        "%B %d, %Y", "%b %d, %Y"
    ]
    clean_str = re.sub(r'(?<=\d)(st|nd|rd|th)', '', date_str)
    for fmt in formats:
        try:
            return datetime.strptime(clean_str, fmt)
        except ValueError:
            continue
    return None


def format_date(dt_obj, original_format_hint):
    day = dt_obj.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    if '-' in original_format_hint:
        return dt_obj.strftime("%d-%m-%Y")
    elif ',' in original_format_hint:
        return dt_obj.strftime(f"%B {day}{suffix}, %Y")
    else:
        return dt_obj.strftime(f"{day}{suffix} %B %Y")


# ==============================================================================
# 3. TIME SHIFT LOGIC
# ==============================================================================

class TimeShifter:
    def __init__(self, original_dates, target_start_str, target_end_str):
        self.target_start = datetime.strptime(target_start_str, "%Y-%m-%d")
        self.target_end = datetime.strptime(target_end_str, "%Y-%m-%d")

        self.orig_start = min(original_dates)
        self.orig_end = max(original_dates)

        self.orig_duration = (self.orig_end - self.orig_start).total_seconds()
        self.target_duration = (self.target_end - self.target_start).total_seconds()

        self.scale = self.target_duration / self.orig_duration if self.orig_duration > 0 else 1

    def shift_date(self, dt_obj):
        if not dt_obj:
            return None
        delta_seconds = (dt_obj - self.orig_start).total_seconds()
        new_date = self.target_start + timedelta(seconds=delta_seconds * self.scale)

        if new_date > self.target_end:
            return self.target_end
        if new_date < self.target_start:
            return self.target_start
        return new_date

    def shift_year(self, year_str):
        try:
            year = int(year_str)
            if year < 2000 or year > 2050:
                return year_str
            new_date = self.shift_date(datetime(year, 7, 1))
            return str(new_date.year)
        except Exception:
            return year_str


# ==============================================================================
# 4. CONTENT PROCESSING
# ==============================================================================

def process_text_content(text, shifter, stats):
    """Shift temporal mentions inside email content while preserving relative chronology."""
    date_pattern = re.compile(
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b'
        r'|\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b'
        r'|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b',
        re.IGNORECASE
    )

    def date_replacer(match):
        original = match.group(0)
        dt = parse_date(original)
        if dt:
            stats.dates_found += 1
            stats.dates_shifted += 1
            return format_date(shifter.shift_date(dt), original)
        return original

    text = date_pattern.sub(date_replacer, text)

    year_pattern = re.compile(r'(?<!\d)(20\d{2})(?!\d)')

    def year_replacer(match):
        original = match.group(1)
        new_year = shifter.shift_year(original)
        if new_year != original:
            stats.years_shifted += 1
        return new_year

    return year_pattern.sub(year_replacer, text)


# ==============================================================================
# 5. MAIN PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess and clean the PATRA email corpus")
    parser.add_argument("--input_file", required=True, help="Path to raw PATRA dataset")
    parser.add_argument("--output_file", required=True, help="Path for cleaned output dataset")
    parser.add_argument("--target_start", default="2019-04-01", help="Target date range start (YYYY-MM-DD, default: 2019-04-01)")
    parser.add_argument("--target_end", default="2025-03-31", help="Target date range end (YYYY-MM-DD, default: 2025-03-31)")
    args = parser.parse_args()

    stats = ProcessingStatistics()

    print(f"Reading {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print("Fixing structural newline issues...")
    content = re.sub(r'EMAIL_END\s*Thread ID:', 'EMAIL_END\n\nThread ID:', content)

    raw_blocks = re.split(r'(?=^Thread ID:)', content, flags=re.MULTILINE)
    raw_blocks = [b for b in raw_blocks if b.strip()]

    stats.total_emails = len(raw_blocks)
    stats.formatting_fixes = stats.total_emails
    print(f"Found {stats.total_emails} email blocks after structural fix.")

    # --------------------------------------------------------------------------
    # PASS 1: PARSE STRUCTURE & DATES
    # --------------------------------------------------------------------------
    emails = []
    all_dates = []

    for block in raw_blocks:
        lines = block.strip().split('\n')
        email_obj = {
            'lines': lines,
            'subject_norm': '',
            'header_date_obj': None,
            'original_thread_id': ''
        }

        for line in lines:
            if line.startswith("Date:"):
                dt = parse_date(line.replace("Date:", "").strip())
                if dt:
                    email_obj['header_date_obj'] = dt
                    all_dates.append(dt)
            elif line.startswith("Subject:"):
                email_obj['subject_norm'] = normalize_subject(line.replace("Subject:", "").strip())
            elif line.startswith("Thread ID:"):
                email_obj['original_thread_id'] = line.replace("Thread ID:", "").strip()

        emails.append(email_obj)

    if not all_dates:
        print("Error: No valid dates found.")
        return

    shifter = TimeShifter(all_dates, args.target_start, args.target_end)
    stats.set_ranges(shifter.orig_start, shifter.orig_end, shifter.target_start, shifter.target_end)

    # --------------------------------------------------------------------------
    # PASS 2: THREAD ID ASSIGNMENT
    # --------------------------------------------------------------------------
    print("Assigning Thread IDs...")

    subject_to_thread_id = {}
    used_ids = set()

    for email in emails:
        normalized = email['subject_norm']
        if normalized:
            stats.unique_normalized_subjects.add(normalized)

        if normalized and normalized in subject_to_thread_id:
            email['new_thread_id'] = subject_to_thread_id[normalized]
            stats.emails_grouped_into_existing_threads += 1
        else:
            while True:
                new_id = generate_thread_id()
                if new_id not in used_ids:
                    break
            used_ids.add(new_id)
            email['new_thread_id'] = new_id
            stats.new_threads_created += 1
            if normalized:
                subject_to_thread_id[normalized] = new_id

        stats.new_unique_thread_ids.add(email['new_thread_id'])

    # --------------------------------------------------------------------------
    # PASS 3: CONTENT TRANSFORMATION & REASSEMBLY
    # --------------------------------------------------------------------------
    print("Transforming content...")
    final_output_blocks = []

    for email in emails:
        new_lines = []

        for line in email['lines']:
            if line.startswith("Thread ID:"):
                new_lines.append(f"Thread ID: {email['new_thread_id']}")
            elif line.startswith("Date:"):
                if email['header_date_obj']:
                    new_dt = shifter.shift_date(email['header_date_obj'])
                    orig_str = line.replace("Date:", "").strip()
                    new_lines.append(f"Date: {format_date(new_dt, orig_str)}")
                    stats.dates_shifted += 1
                else:
                    new_lines.append(line)
            else:
                new_lines.append(process_text_content(line, shifter, stats))

        final_output_blocks.append("\n".join(new_lines))

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(final_output_blocks))

    stats.generate_report()
    print(f"File saved to {args.output_file}")


if __name__ == "__main__":
    main()
