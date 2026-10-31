#!/usr/bin/env python3
"""Private, read-only recovery of PIN-to-state coverage for the 6,039 WSQS cohort.

Reads originals; creates PRIVATE CSV/JSON reports outside the manuscript project.
Never edits survey workbooks, locked figures, repositories or reproducibility archives.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata

from openpyxl import load_workbook

ROOT = Path.home() / 'Desktop/Spring 2026/Chapter 3 MANUAL RERUN'
PRIVATE = Path.home() / 'WSQS_PRIVATE_COVERAGE_N6039_NOT_FOR_PUBLIC'
PRIVATE.mkdir(parents=True, exist_ok=True)
MASTER = ROOT / 'merged_cleaned_survey_data.xlsx'
EXFILE = ROOT / 'WSQS_MODEL_RERUN_EXCLUSION_IDS/WSQS_EXCLUDE_106_PRIMARY_N6039.csv'
LOCAL = (ROOT / 'Demographid analysis/03_PIN_geographic_coverage_CORRECTED'
         / 'Chapter3_STEP03A_R1_PIN_geographic_coverage_CORRECTED.xlsx')
MIRROR_LOCAL = PRIVATE / 'all_india_pincode_mirror_2024.csv'
REFERENCE_SOURCE = '2024 India-wide PIN directory'


def norm_header(s):
    return re.sub(r'[^a-z0-9]', '', str(s or '').lower())


def txt(s):
    if s is None:
        return ''
    if isinstance(s, float) and s.is_integer():
        s = int(s)
    return unicodedata.normalize('NFKC', str(s)).strip()


def digits6(s):
    v = txt(s)
    if re.fullmatch(r'[0-9]{6}(?:\.0+)?', v):
        return v.split('.')[0]
    return None


def state_name(value):
    val = ' '.join(txt(value).replace('_', ' ').split()).strip().title()
    k = norm_header(val)
    common = {
        'tn': 'Tamil Nadu', 'tamilnadu': 'Tamil Nadu',
        'andhrapradesh': 'Andhra Pradesh', 'telangana': 'Telangana',
        'puducherry': 'Puducherry', 'pondicherry': 'Puducherry',
        'nctofdelhi': 'Delhi', 'delhi': 'Delhi',
        'jammuandkashmir': 'Jammu and Kashmir',
        'jammukashmir': 'Jammu and Kashmir',
        'dadraandnagarhavelianddamananddiu': 'Dadra and Nagar Haveli and Daman and Diu',
        'dadraandnagarhaveli': 'Dadra and Nagar Haveli',
        'damananddiu': 'Daman and Diu',
        'odisha': 'Odisha', 'orissa': 'Odisha',
        'uttaranchal': 'Uttarakhand', 'uttarakhand': 'Uttarakhand',
        'chhattisgarh': 'Chhattisgarh', 'chattisgarh': 'Chhattisgarh',
        'northest': '', 'northeast': ''
    }
    return common.get(k, val)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def choose_col(keys, choices):
    for x in choices:
        if x in keys:
            return keys.index(x)
    return None


for source in (MASTER, EXFILE, LOCAL):
    if not source.is_file():
        raise SystemExit('STOP: Required input missing: ' + str(source))

with EXFILE.open(newline='', encoding='utf-8-sig') as f:
    rd = csv.DictReader(f)
    if 'ResponseId' not in (rd.fieldnames or []):
        raise SystemExit('STOP: Exclusion manifest lacks ResponseId')
    excluded_list = [txt(row['ResponseId']) for row in rd]
EX = set(excluded_list)
if len(EX) != len(excluded_list) or len(EX) != 106 or not all(EX):
    raise SystemExit('STOP: Manifest is not 106 unique nonblank IDs')

# Trusted local lookups are specifically the 691-record geographic reference.
local_states = defaultdict(set)
wb = load_workbook(LOCAL, data_only=True, read_only=True)
try:
    ws = wb['09_study_PIN_lookup']
    it = ws.iter_rows(values_only=True)
    headers = [norm_header(x) for x in next(it)]
    pi = choose_col(headers, ['pin', 'pincode', 'postalcode'])
    si = choose_col(headers, ['state', 'statename'])
    if pi is None or si is None:
        raise SystemExit('STOP: PIN/state columns missing from local reference')
    for row in it:
        p = digits6(row[pi] if pi < len(row) else None)
        st = state_name(row[si] if si < len(row) else None)
        if p and st:
            local_states[p].add(st)
finally:
    wb.close()
print('LOCAL REFERENCE:', len(local_states), 'PINs')

# Use the same dated India-wide PIN reference as the published coverage analysis.
# Its source file is supplied privately by the analyst, not copied to this archive.
broad = MIRROR_LOCAL
broad_label = 'ALL_INDIA_PIN_DIRECTORY_2024'
if not broad.is_file():
    raise SystemExit('STOP: Missing 2024 PIN reference at ' + str(broad))

broad_states = defaultdict(set)
with broad.open(newline='', encoding='utf-8-sig', errors='replace') as f:
    reader = csv.reader(f)
    header = next(reader)
    keys = [norm_header(h) for h in header]
    pi = choose_col(keys, ['pincode', 'pin', 'postalcode'])
    si = choose_col(keys, ['statename', 'state', 'stateut'])
    if pi is None or si is None:
        raise SystemExit('STOP: Unrecognized broad directory headers: ' + repr(header[:18]))
    nref = 0
    for row in reader:
        nref += 1
        if len(row) <= max(pi, si):
            continue
        p = digits6(row[pi])
        st = state_name(row[si])
        if p and st:
            broad_states[p].add(st)
if len(broad_states) < 12_000:
    raise SystemExit('STOP: Broad directory too small to classify Indian PINs')
print('ALL-INDIA REFERENCE:', len(broad_states), 'PINs;', nref, 'office records')
print('REFERENCE STATUS:', broad_label)

# Read only 3 survey fields from 6,145 source rows and exclude 106 IDs in memory.
wb = load_workbook(MASTER, data_only=True, read_only=True)
try:
    ws = wb.worksheets[0]
    header_row = None
    id_idx = None
    pin_idx = None
    for idx, row in enumerate(ws.iter_rows(max_row=12, values_only=True), 1):
        keys = [norm_header(x) for x in row]
        if 'responseid' in keys and 'q151' in keys:
            header_row = idx
            id_idx = keys.index('responseid')
            pin_idx = keys.index('q151')
            break
    if header_row is None:
        raise SystemExit('STOP: Cannot find ResponseId and Q 15.1 in master')
    original_seen = set()
    removed = set()
    cohort = []
    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        rid = txt(row[id_idx] if id_idx < len(row) else None)
        if not rid:
            continue
        if rid in original_seen:
            raise SystemExit('STOP: Duplicate survey ResponseId')
        original_seen.add(rid)
        if rid in EX:
            removed.add(rid)
        else:
            cohort.append((rid, row[pin_idx] if pin_idx < len(row) else None))
finally:
    wb.close()

if not (len(original_seen) == 6145 and removed == EX and len(cohort) == 6039):
    raise SystemExit('STOP: 6145 -> 6039 cohort verification failed')


def candidates(raw):
    s = txt(raw)
    if not s:
        return [], 'MISSING'
    if re.fullmatch(r'\d{6}', s):
        return [s], 'EXACT'
    if re.fullmatch(r'\d{6}\.0+', s):
        return [s.split('.')[0]], 'NUMERIC_FORMAT'
    no_punctuation = re.sub(r'[\s\u00a0,.;:/\\()\-\u2013\u2014]+', '', s)
    if re.fullmatch(r'\d{6}', no_punctuation):
        return [no_punctuation], 'PUNCTUATION_FORMAT'
    # Six consecutive digits embedded in text, or grouped as XXX XXX.
    # Never take a substring of a longer number (e.g. phone number).
    parts = re.findall(r'(?<!\d)(?:[0-9]{6}|[0-9]{3}[\s\-]+[0-9]{3})(?!\d)', s)
    parts = list(dict.fromkeys(re.sub(r'\D', '', x) for x in parts))
    if len(parts) == 1:
        return parts, 'EMBEDDED_UNIQUE_PIN'
    if len(parts) > 1:
        return parts, 'MULTIPLE_PINS_REVIEW'
    return [], 'INVALID_INCOMPLETE_OR_FREE_TEXT'


def classify(p):
    a = set(local_states.get(p, []))
    b = set(broad_states.get(p, []))
    if len(a) > 1 or len(b) > 1:
        return None, 'MULTIPLE_STATES_IN_REFERENCE'
    if a and b and a != b:
        return None, 'REFERENCE_STATE_CONFLICT'
    if a:
        return next(iter(a)), 'LOCAL_AND_WIDE' if b else 'LOCAL_REFERENCE'
    if b:
        return next(iter(b)), 'INDIA_WIDE_DIRECTORY'
    return None, 'NO_POSTAL_DIRECTORY_MATCH'

state_counts = Counter()
status_counts = Counter()
normalization_counts = Counter()
private_rows = []
review_rows = []
linked = 0

for rid, raw in cohort:
    pins, method = candidates(raw)
    normalization_counts[method] += 1
    state = None
    pin_used = ''
    source = ''
    status = ''
    if len(pins) == 1:
        pin_used = pins[0]
        state, source = classify(pin_used)
        if state:
            status = 'STATE_VERIFIED'
            state_counts[state] += 1
            linked += 1
        else:
            status = source
    elif len(pins) > 1:
        status = 'AMBIGUOUS_MULTIPLE_PINS'
    else:
        status = method
    status_counts[status] += 1
    rec = [rid, txt(raw), pin_used, method, state or '', source, status]
    private_rows.append(rec)
    if not state:
        review_rows.append(rec)

unresolved = len(cohort) - linked
assert sum(state_counts.values()) == linked
assert linked + unresolved == 6039
out_tn = linked - state_counts['Tamil Nadu']

# Output only PRIVATE record-level files, never in public reproduction folder.
with (PRIVATE / 'S4_PRIVATE_PIN_RECOVERY_RESPONSEIDS.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['ResponseId', 'Raw_PIN', 'Normalized_PIN', 'Normalization', 'State',
                'Evidence', 'Status'])
    w.writerows(private_rows)
with (PRIVATE / 'S4_PRIVATE_UNRESOLVED_PIN_REVIEW.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['ResponseId', 'Raw_PIN', 'Candidate_PIN', 'Normalization', 'State',
                'Evidence', 'Status'])
    w.writerows(review_rows)

with (PRIVATE / 'S4_AGGREGATE_STATE_COUNTS_N6039.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['State_or_status', 'Households', 'Percent_of_all_6039',
                'Percent_of_state_classified', 'Percent_of_outside_TN_only'])
    for state, n in sorted(state_counts.items(), key=lambda x: (-x[1], x[0])):
        w.writerow([state, n, round(100*n/6039, 4),
                    round(100*n/linked, 4) if linked else '',
                    round(100*n/out_tn, 4) if out_tn and state != 'Tamil Nadu' else ''])
    w.writerow(['UNRESOLVED', unresolved, round(100*unresolved/6039, 4), '', ''])
    w.writerow(['TOTAL_ANALYTICAL', 6039, 100, '', ''])

with (PRIVATE / 'S4_AGGREGATE_OUTSIDE_TN_BY_STATE_N6039.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['State_or_UT', 'Households', 'Percent_of_all_6039',
                'Percent_of_confirmed_outside_TN'])
    for st, n in sorted(state_counts.items(), key=lambda x: (-x[1], x[0])):
        if st != 'Tamil Nadu':
            w.writerow([st, n, round(100*n/6039, 4), round(100*n/out_tn, 4)])
    w.writerow(['TOTAL_OUTSIDE_TN_CLASSIFIED', out_tn,
                round(100*out_tn/6039, 4), 100 if out_tn else 0])

report = {
    'run_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PIN_DIRECTORY_2024',
    'N_original': 6145, 'excluded_ResponseIds': 106, 'analytical_N': 6039,
    'state_classified': linked, 'tamil_nadu': state_counts['Tamil Nadu'],
    'outside_tamil_nadu_classified': out_tn, 'unresolved': unresolved,
    'broad_reference': str(broad), 'broad_reference_label': broad_label,
    'broad_reference_sha256': sha(broad), 'reference_source': REFERENCE_SOURCE,
    'local_reference_sha256': sha(LOCAL),
    'normalization_method_counts': dict(normalization_counts),
    'classification_status_counts': dict(status_counts),
    'states': dict(sorted(state_counts.items())),
    'figure1_modified': False, 'reproducibility_archive_modified': False,
    'interpretation': 'State of reported postal PIN; not independently GPS-verified residence.'
}
(PRIVATE / 'S4_AGGREGATE_STATE_AUDIT.json').write_text(json.dumps(report, indent=2))

print('\n' + '='*78)
print('WSQS S4 | RECOVERED PIN-TO-STATE COVERAGE | ANALYTICAL N=6039')
print('='*78)
print('State-classified: ', f'{linked:,}', f'({linked/6039:.2%} of all)')
print('Tamil Nadu:       ', f'{state_counts["Tamil Nadu"]:,}',
      f'({state_counts["Tamil Nadu"]/6039:.2%} of all)')
print('Outside Tamil Nadu:', f'{out_tn:,}', f'({out_tn/6039:.2%} of all)')
print('Unresolved:       ', f'{unresolved:,}', f'({unresolved/6039:.2%} of all)')
print('\nSTATES OUTSIDE TAMIL NADU | share of all N=6039 and outside-TN subset')
print(f'{"State / UT":<38}{"N":>6}{"% all":>10}{"% outside":>12}')
for st, n in sorted(state_counts.items(), key=lambda x: (-x[1], x[0])):
    if st == 'Tamil Nadu':
        continue
    print(f'{st:<38}{n:>6}{100*n/6039:>10.2f}{100*n/out_tn:>12.2f}')
print('\nNORMALIZATION METHODS')
for k,v in normalization_counts.most_common():
    print(f' {k:<35} {v:>5}')
print('\nREVIEW STATUS COUNTS')
for k,v in status_counts.most_common():
    print(f' {k:<35} {v:>5}')
print('\nUnresolved response counts by category:', dict(Counter(r[-1] for r in review_rows)))
print('Individual-level review entries are saved ONLY in the local private directory.')
print('\nPRIVATE OUTPUT DIRECTORY:', PRIVATE)
print('STATE TABLE: S4_AGGREGATE_STATE_COUNTS_N6039.csv')
print('OUTSIDE-TN:  S4_AGGREGATE_OUTSIDE_TN_BY_STATE_N6039.csv')
print('UNRESOLVED:  S4_PRIVATE_UNRESOLVED_PIN_REVIEW.csv')
print('\nLOCKED FIGURE 1: UNTOUCHED. REPRODUCIBILITY PACKAGES: UNTOUCHED.')
print('POSTAL REFERENCE: 2024 India-wide PIN directory.')
print('='*78)
