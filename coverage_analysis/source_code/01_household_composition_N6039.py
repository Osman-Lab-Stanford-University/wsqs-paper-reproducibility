#!/usr/bin/env python3
"""Verified N6039 aggregate harmonization template (housing-tenure bug corrected).
Reads frozen sources, writes only private aggregate files; does not touch figures.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv, json, hashlib, re, unicodedata
from openpyxl import load_workbook

ROOT = Path.home() / 'Desktop/Spring 2026/Chapter 3 MANUAL RERUN'
DEM = ROOT / 'Demographid analysis'
MASTER = ROOT / 'merged_cleaned_survey_data.xlsx'
EXFILE = ROOT / 'WSQS_MODEL_RERUN_EXCLUSION_IDS/WSQS_EXCLUDE_106_PRIMARY_N6039.csv'
OUT = ROOT / 'S4_N6039_COVERAGE_REPRODUCE_NEW_OUTPUT_ONLY'
PRIVATE = Path.home() / 'WSQS_PRIVATE_COVERAGE_N6039_NOT_FOR_PUBLIC'
for p in (MASTER, EXFILE, DEM):
    if not p.exists():
        raise SystemExit(f'STOP: Required source missing: {p}')
if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(f'STOP: Prior output preserved: {OUT}\nRename that folder before repeating.')
OUT.mkdir(exist_ok=True)

def txt(v):
    if v is None: return ''
    if isinstance(v, float) and v.is_integer(): v = int(v)
    return ' '.join(unicodedata.normalize('NFKC', str(v)).split()).strip()

def key(v):
    return re.sub('[^a-z0-9]', '', txt(v).lower())

def fingerprint(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

with EXFILE.open(newline='',encoding='utf-8-sig') as f:
    rd=csv.DictReader(f)
    if 'ResponseId' not in (rd.fieldnames or []):
        raise SystemExit('STOP: Manifest has no ResponseId')
    ids=[txt(r['ResponseId']) for r in rd]
EX=set(ids)
if len(EX)!=106 or len(ids)!=106 or '' in EX:
    raise SystemExit('STOP: Exclusion manifest is not 106 unique nonempty IDs')

sizes=Counter(); tenure=Counter(); cars=Counter(); provision=Counter()
unknown={'size':Counter(),'tenure':Counter(),'cars':Counter(),'provision':Counter()}
seen=set(); removed=set()

wb = load_workbook(MASTER,read_only=True,data_only=True)
try:
    ws=wb.active
    row_iterator=ws.iter_rows(values_only=True)
    headers=None
    for head_idx,row in enumerate(row_iterator,1):
        cols={key(v):i for i,v in enumerate(row) if v is not None}
        if {'responseid','q1510','q1515','q1513','q11'}.issubset(cols):
            headers=cols
            break
        if head_idx>=12: break
    if headers is None: raise SystemExit('STOP: Required headers not found')
    for row in row_iterator:
        rid=txt(row[headers['responseid']])
        if not rid: continue
        if rid in seen: raise SystemExit('STOP: Duplicate ResponseId in source')
        seen.add(rid)
        if rid in EX:
            removed.add(rid)
            continue
        def read(k): return txt(row[headers[k]] if len(row)>headers[k] else None)
        s=read('q1510')
        if s in {'8 or more','8 அல்லது அதற்கு மேற்பட்டவை'}:
            sizes['7+']+=1; sizes['8+ detail']+=1
        elif s in {'7','6','5','4','3','2','1'}:
            sizes[s]+=1
            if s=='7': sizes['7+']+=1
        elif not s: sizes['MISSING']+=1
        else: unknown['size'][s]+=1

        s=read('q1515'); low=s.lower()
        # Classify *free-and-clear* BEFORE mortgage/loan: the original
        # English text can contain references to loans in its definition.
        if ('free and clear' in low or
            low.startswith('owned by you or someone in this household free') or
            s.startswith('உங்களுக்கோ')):
            tenure['Owned outright']+=1
        elif (low.startswith('owned by you or someone in this household with a') or
              s.startswith('அடமானம் அல்லது கடன் மூலம்')):
            tenure['Owned mortgage/loan']+=1
        elif low=='rented' or 'வாடகைக்கு' in s:
            tenure['Rented']+=1
        elif low=='other' or 'மற்றவை' in s:
            tenure['Other']+=1
        else: unknown['tenure'][s or '(MISSING)']+=1

        s=read('q1513')
        if not s: cars['MISSING']+=1
        elif s.lower()=='none' or s=='எதுவும் இல்லை': cars['No car (explicit)']+=1
        elif s in {'1','2','3','4','5'}: cars['Has car']+=1
        else: unknown['cars'][s]+=1

        s=read('q11'); low=s.lower()
        if low=='water from municipality/corporation' or s=='நகராட்சி/மாநகராட்சியிலிருந்து தண்ணீர்':
            provision['Municipality/Corporation']+=1
        elif low=='groundwater' or s=='நிலத்தடி நீர்':
            provision['Groundwater']+=1
        elif low=='lake/pond' or s=='ஏரி/குளம்':
            provision['Lake/Pond']+=1
        elif low=='other sources' or s=='பிற ஆதாரங்கள்':
            provision['Other']+=1
        elif low=='buy water from trucks' or s=='வாகனங்களிலிருந்து தண்ணீர் வாங்கவும்':
            provision['Tanker/Truck']+=1
        elif low=='buy water from stores (e.g., canned water)' or s=='கடைகளில் தண்ணீர் வாங்கவும் (எ.கா., கேன் தண்ணீர்)':
            provision['Commercial/Canned']+=1
        else: unknown['provision'][s]+=1

finally:
    wb.close()

assert len(seen)==6145 and removed==EX and len(seen)-len(removed)==6039, 'STOP: Cohort mismatch'
if any(unknown[k] for k in unknown):
    print('STOP: Unknown categories (check exact raw values):')
    for k,values in unknown.items():
        if values: print(k,dict(values))
    raise SystemExit(2)
assert sizes['MISSING']==26 and sum(sizes[str(i)] for i in range(1,7))+sizes['7+']==6013
assert sum(tenure.values())==6039 and tenure['Owned outright']+tenure['Owned mortgage/loan']==4058
assert tenure['Owned outright']==2691 and tenure['Owned mortgage/loan']==1367, 'STOP: verified tenure subclasses do not match'
assert sum(cars.values())==6039 and cars['Has car']==1464 and cars['MISSING']==2668
assert dict(provision)=={'Municipality/Corporation':2848,'Groundwater':2630,'Lake/Pond':322,'Other':94,'Tanker/Truck':81,'Commercial/Canned':64}

size_order=['1','2','3','4','5','6','7+']
with (OUT/'TABLE_S4_HOUSEHOLD_SIZE_N6039.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['Size_category','N','Percent_of_valid_6013','Percent_of_all_6039'])
    for label in size_order:
        n=sizes[label]; w.writerow([label,n,round(n/6013*100,4),round(n/6039*100,4)])
    w.writerow(['MISSING',26,'',round(26/6039*100,4)])

with (OUT/'TABLE_S4_TENURE_N6039.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['Tenure','N','Percent_of_6039'])
    for label,n in [('Owned outright',tenure['Owned outright']),('Owned mortgage/loan',tenure['Owned mortgage/loan']),('Owned total',4058),('Rented',tenure['Rented']),('Other',tenure['Other'])]:
        w.writerow([label,n,round(n/6039*100,4)])

with (OUT/'TABLE_S4_CAR_MISSINGNESS_N6039.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['Car_response','N','Percent_of_all_6039','Percent_of_explicit_3371'])
    for label in ['Has car','No car (explicit)','MISSING']:
        n=cars[label]; w.writerow([label,n,round(n/6039*100,4),round(n/3371*100,4) if label!='MISSING' else ''])

with (OUT/'TABLE_S4_WATER_PROVISION_N6039.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['Provision','N','Percent_of_6039'])
    for label,n in sorted(provision.items(), key=lambda t:-t[1]):
        w.writerow([label,n,round(n/6039*100,4)])

# Original coverage figure's binary operational recode. Distinguish source
# English-form blanks from explicit Tamil "None" in the separate missingness CSV.
operational_no_car = cars['No car (explicit)'] + cars['MISSING']
assert operational_no_car == 4575 and operational_no_car + cars['Has car'] == 6039
with (OUT/'TABLE_S4_CAR_BINARY_HISTORICAL_N6039.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['Category','N','Percent_of_6039','Coding_basis'])
    w.writerow(['No car',operational_no_car,round(100*operational_no_car/6039,4),
                'Historical coverage convention: English blank included in No car'])
    w.writerow(['One or more cars',cars['Has car'],round(100*cars['Has car']/6039,4),
                'Explicit one or more car response'])

ref_manifest=[]
# Discover exact source sheets/field names without guessing the cached spatial schema.
refpaths=[
    DEM/'02_household_size_benchmark/Chapter3_STEP02A_household_size_benchmark.xlsx',
    DEM/'03_PIN_geographic_coverage_CORRECTED/Chapter3_STEP03A_R1_PIN_geographic_coverage_CORRECTED.xlsx',
    DEM/'STEP02B_HH_SIZE_2025_SPATIAL_PROJECTION/STEP02B_household_size_projection_results.xlsx',
    DEM/'01_sample_coverage_audit/Chapter3_sample_coverage_audit_STEP01.xlsx',
]
for p in refpaths:
    if not p.is_file(): continue
    wb=load_workbook(p,read_only=True,data_only=True)
    try:
        for ws in wb.worksheets:
            previews=[]
            for row in ws.iter_rows(max_row=min(8,ws.max_row),values_only=True):
                if any(v is not None and str(v).strip() for v in row):
                    previews.append([txt(x)[:65] for x in row[:16]])
                if len(previews)==2: break
            ref_manifest.append({'file':str(p.relative_to(DEM)),'sheet':ws.title,
                                 'approx_rows':ws.max_row,'columns':ws.max_column,
                                 'header_candidates':previews})
    finally: wb.close()

report={
 'status':'AGGREGATES_VERIFIED_REFERENCES_PENDING_RECALC',
 'original_n':6145,'excluded':106,'revised_n':6039,
 'household_size_valid':6013,'household_size_missing':26,
 'household_size_7plus':sizes['7+'],'household_size_8plus_detail':sizes['8+ detail'],
 'tenure':dict(tenure),'car_response':dict(cars),'provision':dict(provision),
 'car_rate_all':100*cars['Has car']/6039,
 'car_rate_among_explicit':100*cars['Has car']/3371,
 'car_missing_fraction':100*cars['MISSING']/6039,
 'master_sha256':fingerprint(MASTER),
 'exclusion_manifest_sha256':fingerprint(EXFILE),
 'figure1_modified':False,'public_package_modified':False,
 'benchmark_reference_data_not_modified':True,
}
(OUT/'S4_N6039_aggregate_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(OUT/'S4_REFERENCE_WORKBOOK_SHEET_MAP.json').write_text(json.dumps(ref_manifest,indent=2),encoding='utf-8')

print('\n'+'='*72)
print('S4 | N6039 HARMONIZED HOUSEHOLD COVERAGE | PASS')
print('='*72)
print('Original 6145 -> removed 106 -> revised 6039')
print('HOUSEHOLD SIZE: valid=6013, missing=26, 7+=',sizes['7+'])
for s in size_order: print(f'  {s:>2}: {sizes[s]:4d} | valid % {100*sizes[s]/6013:.2f}')
print('TENURE:', dict(tenure),'| TOTAL OWNED',4058,'|',f'{4058/6039:.2%}')
print('CAR: reported >=1',cars['Has car'],f'({cars["Has car"]/6039:.2%} all)',
      '| explicit none',cars['No car (explicit)'],
      '| blank',cars['MISSING'],
      '| affirmative among explicit',f'{cars["Has car"]/3371:.2%}')
print('PROVISION:',dict(provision))
print('\nREFERENCE SHEET INVENTORY (for exact benchmarking and GHS spatial rerun)')
for r in ref_manifest:
    headers=r['header_candidates']
    head=' | '.join(headers[0])[:210] if headers else '[blank]'
    print('  '+r['file'].split('/')[-1], '|',r['sheet'], '| rows',r['approx_rows'],'|',head)
print('\nAGGREGATE OUTPUTS:',OUT)
print('LOCKED FIGURE 1: UNTOUCHED')
print('PUBLIC REPRODUCIBILITY FILES: UNTOUCHED')
print('='*72)
