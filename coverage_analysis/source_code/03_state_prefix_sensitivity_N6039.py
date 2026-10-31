#!/usr/bin/env python3
"""PRIVATE WSQS S4: assess state-only recovery without changing any respondent PIN.

Uses the completed private 6,039-record PIN audit and the broad postal directory.
Exact postal matches and state-only prefix inferences are NEVER conflated.
Writes only to ~/WSQS_PRIVATE_COVERAGE_N6039_NOT_FOR_PUBLIC.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv, json, re, unicodedata

HOME = Path.home()
PRIVATE = HOME / 'WSQS_PRIVATE_COVERAGE_N6039_NOT_FOR_PUBLIC'
ROWFILE = PRIVATE / 'S4_PRIVATE_PIN_RECOVERY_RESPONSEIDS.csv'
MIRROR = PRIVATE / 'all_india_pincode_mirror_2024.csv'
REF = MIRROR
if not ROWFILE.is_file() or not REF.is_file():
    raise SystemExit('STOP: First run S4_PIN_RECOVERY_AND_STATE_TABLE_N6039.py; private outputs or postal directory are missing.')

def key(x):
    return re.sub('[^a-z0-9]', '', str(x or '').lower())

def state(x):
    t = ' '.join(unicodedata.normalize('NFKC', str(x or '')).strip().replace('_',' ').split()).title()
    aliases = {'Tamilnadu':'Tamil Nadu','Andhrapradesh':'Andhra Pradesh',
               'Jammu & Kashmir':'Jammu and Kashmir','Jammu And Kashmir':'Jammu and Kashmir',
               'Pondicherry':'Puducherry', 'Orissa':'Odisha','Uttaranchal':'Uttarakhand'}
    return aliases.get(t,t)

# Strict postal-directory field recognition: do not treat postal circle as state.
prefix_codes = defaultdict(lambda: defaultdict(set))
with REF.open(newline='', encoding='utf-8-sig', errors='replace') as f:
    rd = csv.DictReader(f)
    if not rd.fieldnames:
        raise SystemExit('STOP: Empty postal-directory file')
    col = {key(x):x for x in rd.fieldnames}
    pinfield = next((col[k] for k in ('pincode','pin','postalcode') if k in col),None)
    stfield = next((col[k] for k in ('statename','state','stateut') if k in col),None)
    if not pinfield or not stfield:
        raise SystemExit('STOP: Postal reference requires PIN and geographic state, not CircleName. Found headers: '+repr(rd.fieldnames))
    for r in rd:
        p = str(r.get(pinfield,'')).strip()
        s = state(r.get(stfield,''))
        if re.fullmatch(r'[0-9]{6}',p) and s and key(s) not in {'northeast','northest','unknown'}:
            prefix_codes[p[:3]][s].add(p)

# Only one geographic state allowed for a given three-digit prefix;
# require >=5 DISTINCT published six-digit codes as supporting evidence.
strong = {}
for pref, states in prefix_codes.items():
    if len(states)==1:
        st, codes = next(iter(states.items()))
        if len(codes) >= 5:
            strong[pref] = (st, len(codes))

rows = []
with ROWFILE.open(newline='',encoding='utf-8-sig') as f:
    rd=csv.DictReader(f)
    needed = {'ResponseId','Raw_PIN','Normalized_PIN','Normalization','State','Evidence','Status'}
    if not needed.issubset(rd.fieldnames or []):
        raise SystemExit('STOP: Private recovery output missing expected columns: '+repr(rd.fieldnames))
    rows=list(rd)
if len(rows)!=6039 or len({r['ResponseId'] for r in rows})!=6039:
    raise SystemExit('STOP: Expected exactly 6,039 unique household rows')

matched = Counter()
inferred = Counter()
unresolved = Counter()
inferred_details=[]
review_details=[]
method_counts=Counter()

# A missing PIN can never be recovered from a prefix.
for r in rows:
    st = state(r['State']) if r['State'].strip() else ''
    if st and r['Status']=='STATE_VERIFIED':
        matched[st] += 1
        continue
    p = r['Normalized_PIN'].strip()
    status = r['Status']
    if status=='NO_POSTAL_DIRECTORY_MATCH' and re.fullmatch(r'[0-9]{6}',p):
        info=strong.get(p[:3])
        if info is not None:
            inferred[info[0]] += 1
            method_counts['THREE_DIGIT_PREFIX_SINGLE_STATE'] += 1
            inferred_details.append([r['ResponseId'],p,p[:3],info[0],info[1],
                                     'STATE_ONLY_INFERENCE_NOT_PIN_VERIFICATION'])
            continue
    unresolved[status] += 1
    # For 5- and 7-digit numeric strings, show candidate postal codes to check manually.
    raw = r['Raw_PIN'].strip()
    digits = re.sub(r'\D','',raw)
    if status=='INVALID_INCOMPLETE_OR_FREE_TEXT' and len(digits) in (5,7):
        guesses = set()
        if len(digits)==7:
            for i in range(7): guesses.add(digits[:i]+digits[i+1:])
        else:
            for i in range(6):
                for d in '0123456789':
                    guesses.add(digits[:i]+d+digits[i:])
        # Postal existence based on complete directory (not prefix only).
        good = sorted(q for q in guesses if re.fullmatch(r'[0-9]{6}',q)
                      and q[:3] in prefix_codes and any(q in codes for codes in prefix_codes[q[:3]].values()))
        review_details.append([r['ResponseId'],raw,len(good),';'.join(good[:12]),
                               'MANUAL_REVIEW_ONLY_NO_AUTOMATIC_CORRECTION'])

totmatched=sum(matched.values())
totinferred=sum(inferred.values())
totunresolved=sum(unresolved.values())
assert totmatched + totinferred + totunresolved == 6039

with (PRIVATE/'S4_STATE_COUNTS_WITH_SEPARATE_PREFIX_SENSITIVITY.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['State_UT','Directory_matched_N','Prefix_inferred_state_only_N',
                'Combined_sensitivity_N','Directory_matched_pct_all_6039',
                'Combined_sensitivity_pct_all_6039'])
    for st in sorted(set(matched)|set(inferred),key=lambda s:-(matched[s]+inferred[s])):
        m,i=matched[st],inferred[st]
        w.writerow([st,m,i,m+i,round(m*100/6039,3),round((m+i)*100/6039,3)])
    w.writerow(['UNRESOLVED',0,0,totunresolved,0,round(totunresolved*100/6039,3)])

with (PRIVATE/'S4_PRIVATE_PREFIX_ONLY_STATE_INFERENCE_RESPONSEIDS.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['ResponseId','Reported_6digit_PIN','Prefix3','Inferred_state','Distinct_directory_codes_supporting_prefix','Caution'])
    w.writerows(inferred_details)

with (PRIVATE/'S4_PRIVATE_5_OR_7_DIGIT_CANDIDATES_MANUAL_REVIEW.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['ResponseId','Original_entry','Number_of_directory_matched_edit_candidates',
                'First_12_candidates','Caution'])
    w.writerows(review_details)

print('\n'+'='*76)
print('WSQS S4 | INDEPENDENT STATE-ONLY PREFIX SENSITIVITY | N=6039')
print('='*76)
print('POSTAL REFERENCE:',REF)
print('REFERENCE TYPE: India-wide PIN directory, 2024')
print('Sufficiently supported state-exclusive 3-digit prefixes:',len(strong))
print(f'Directory-matched state records: {totmatched:,} ({totmatched/6039:.2%})')
print(f'Additional STATE-ONLY prefix inferences: {totinferred:,} ({totinferred/6039:.2%})')
print(f'Combined state-classified SENSITIVITY: {totmatched+totinferred:,} ({(totmatched+totinferred)/6039:.2%})')
print(f'Still unresolved: {totunresolved:,} ({totunresolved/6039:.2%})')
print('\nSTATE / UT                      DIRECTORY   PREFIX-ONLY  COMBINED')
for st in sorted(set(matched)|set(inferred),key=lambda s:-(matched[s]+inferred[s])):
    print(f'{st[:30]:30} {matched[st]:>7,} {inferred[st]:>12,} {matched[st]+inferred[st]:>9,}')
print('\nUNRESOLVED REASONS:',dict(unresolved))
print('Manual 5/7-digit candidate records:',len(review_details))
print('Exact-one-directory-candidate cases:',sum(a[2]==1 for a in review_details))
print('\nAll reports PRIVATE:',PRIVATE)
print('No source data, Figure 1, or public reproducibility package modified.')
print('CAUTION: Prefix inference DOES NOT validate the full PIN or household location.')
print('='*76)
