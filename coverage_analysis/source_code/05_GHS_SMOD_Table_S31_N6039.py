#!/usr/bin/env python3
"""S4 Stage7: validate paper-ready S31 and discover/recover exact GHS raster diagnostics.
Read-only inputs. Never changes original workbook, SVG, existing S30-S32 or public repo.
Creates an S31 publication file ONLY if all 26 rows can be verified.
"""
from pathlib import Path
from collections import Counter
import csv, json, re, math, hashlib, unicodedata
from openpyxl import load_workbook

ROOT=Path.cwd()
DEM=ROOT/'Demographid analysis'
PUB=ROOT/'S4_N6039_PUBLICATION_STAGE6'
PRIV=Path.home()/'WSQS_PRIVATE_COVERAGE_N6039_NOT_FOR_PUBLIC'
S4=PRIV/'STAGE4'
OUT=ROOT/'S4_N6039_TABLE_FINALIZATION_STAGE7'
OUT.mkdir(exist_ok=True)

def die(x): raise SystemExit('STOP: '+x)
def norm(x): return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKC',str(x or '')).lower())
def values_csv(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def save_csv(p,header,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

auditpath=PUB/'STAGE6_RECONSTRUCTION_AUDIT.json'
s31path=PUB/'TABLE_S31_N6039_REVIEW.csv'
geopath=S4/'S4_GEOGRAPHIC_FIGURE_INPUT_PRIVATE_N6039.csv'
rosterpath=DEM/'03_PIN_geographic_coverage_CORRECTED/Chapter3_STEP03A_R1_PIN_geographic_coverage_CORRECTED.xlsx'
for path in [auditpath,s31path,geopath,rosterpath]:
    if not path.exists():die('Missing '+str(path))
audit=json.loads(auditpath.read_text())
if audit['N']!=6039 or audit['spatial_N']!=5778 or audit['unique_spatial_PINs']!=629:
    die('Cohort mismatch')
unique=int(audit['unique_normalized_six_digit_PINs'])
if not 629<=unique<=5991:die('Invalid normalized PIN count')
s31=values_csv(s31path)
if len(s31)!=26:die('Expected 26 S31 diagnostic rows')
bylabel={r['Diagnostic']:r for r in s31}
keypin='Unique valid-format PIN codes'
if keypin not in bylabel:die('Missing original S31 PIN row')
if re.sub(r'[^0-9]','',bylabel[keypin]['Revised result'])!=str(unique):
    die('Stage6 unique PIN count inconsistent with S31')

g=values_csv(geopath)
if len(g)!=629 or sum(int(r['Household_N']) for r in g)!=5778:die('629-PIN spatial source inconsistent')
pins={r['PIN'].strip() for r in g}
if len(pins)!=629:die('Repeated source PINs')
districts={norm(r['District']) for r in g}
if len(districts)!=37:die('Expected 37 distinct districts')

wb=load_workbook(rosterpath,read_only=True,data_only=True)
if '04_current38' not in wb.sheetnames:die('Missing authoritative local 38-district roster')
rs=list(wb['04_current38'].iter_rows(values_only=True))
wb.close()
choices=[]
for j in range(max(len(r) for r in rs)):
    v=[norm(r[j]) for r in rs[1:] if j<len(r) and r[j] is not None]
    vv=set(v)
    matches=len(vv & districts)
    if matches>=34 and len(vv)==38:
        choices.append((matches,j,vv))
if not choices:die('38-district column not identifiable; printing its first five rows next')
choices.sort(reverse=True)
_,j,full=choices[0]
missing=sorted(full-districts)
if len(missing)!=1 or missing[0]!='chengalpattu':
    die('Missing district not exclusively Chengalpattu: '+str(missing))

print('='*78)
print('STAGE 7 | PAPER TABLE S31 | INDEPENDENT COHORT CHECK')
print('='*78)
print('Analytical N:',audit['N'])
print('Normalized valid PIN records:',audit['normalized_six_digit_N'])
print('UNIQUE VALID-FORMAT PIN CODES:',unique)
print('Tamil Nadu mapped N:',audit['spatial_N'],'| PINs:',len(pins))
print('Roster compared: 38 districts | represented: 37 | absent: Chengalpattu')

# Candidate sources only. Avoid reading survey row-level files or emitting PIN IDs.
SEARCH_TERMS=('smod','settle','urban','raster','spatial','gis','geocode','geo','pin','density','ghs')
CAND=[]
for p in sorted(DEM.rglob('*')):
    if not p.is_file() or p.suffix.lower() not in ('.csv','.xlsx','.xlsm','.parquet'):
        continue
    if not any(t in p.name.lower() or t in str(p.parent.name).lower() for t in SEARCH_TERMS):
        continue
    if p.stat().st_size>80_000_000: continue
    CAND.append(p)

hints=[]
possible_tables=[]
def consider(path,sheet,headers,preview):
    hh=[norm(z) for z in headers]
    # Avoid survey-only spreadsheets with 100s of columns.
    if len(hh)>100:return
    pin_indices=[i for i,h in enumerate(hh) if h in ('pin','pincode','postalcode','matchedpin','studypin','pin6','cleanedpin')]
    raster_indices=[i for i,h in enumerate(hh) if any(t in h for t in ('smod','settlement','degurba','exactcell','modalclass','neighbourhoodmode','neighborhoodmode'))]
    if raster_indices:
        hints.append((str(path.relative_to(DEM)),sheet,headers[:60],len(pin_indices),len(raster_indices)))
    if pin_indices and raster_indices:
        possible_tables.append((path,sheet,headers,pin_indices,raster_indices))

for p in CAND:
    try:
        if p.suffix.lower()=='.csv':
            with p.open(newline='',encoding='utf-8-sig') as f:
                rd=csv.reader(f); head=next(rd,[])
                consider(p,'CSV',head,[])
        elif p.suffix.lower() in ('.xlsx','.xlsm'):
            wb=load_workbook(p,read_only=True,data_only=True)
            for ws in wb:
                if ws.max_column and ws.max_column>100:continue
                it=ws.iter_rows(min_row=1,max_row=3,values_only=True)
                head=list(next(it,[]));consider(p,ws.title,head,[])
            wb.close()
        # parquet not opened without pandas; report filename only in discovery text
    except Exception as exc:
        print('SKIPPED UNREADABLE:',p.name,type(exc).__name__)

print('\nRASTER SOURCE DISCOVERY (column names only; no survey rows):')
for path,sheet,headers,pinn,raw in hints[:25]:
    useful=[str(h) for h in headers if h is not None and any(k in norm(h) for k in ('pin','smod','class','urban','cell','mode','settle','neigh'))]
    print('  ',path,'|',sheet,'| PIN column:',bool(pinn))
    print('     ', ' ; '.join(useful)[:350])
if not hints:
    print('  No detailed machine-readable classification fields found in candidate Excel/CSV files.')
print('Eligible PIN-linked raster-source tables:',len(possible_tables))
rasters=[p for p in DEM.rglob('*') if p.is_file() and p.suffix.lower() in ('.tif','.tiff') and ('smod' in p.name.lower() or 'ghs' in p.name.lower())]
print('Local GHS raster candidates:',len(rasters))
for p in rasters[:12]:print('  ',p.relative_to(DEM))
if not possible_tables and not rasters:
    print('No retained-PIN detailed class/3x3 source identified locally; do not carry historical values forward.')

# Auto-recover only from unmistakable columns that contain at least 629 retained PIN codes.
GHSCODES={10,11,12,13,21,22,23,30}
results=None
for path,sheet,headers,pin_i,raster_i in possible_tables:
    hh=[norm(z) for z in headers]
    exact_candidates=[i for i in raster_i if hh[i] == 'smodexactcode']
    mode_candidates=[i for i in raster_i if hh[i] == 'smod3x3modecode']
    if len(exact_candidates)!=1 or len(mode_candidates)!=1 or len(pin_i)!=1:continue
    xcol,mcol,pcol=exact_candidates[0],mode_candidates[0],pin_i[0]
    dat={}
    def add(row):
        if len(row)<=max(xcol,mcol,pcol):return
        p=str(row[pcol] or '').strip()
        if re.fullmatch(r'\d{6}\.0',p):p=p[:-2]
        if p not in pins:return
        try:exact=int(float(row[xcol]));mode=int(float(row[mcol]))
        except (TypeError,ValueError):return
        if exact not in GHSCODES or mode not in GHSCODES:return
        if p in dat and dat[p]!=(exact,mode):die('Conflicting raster values for PIN '+p)
        dat[p]=(exact,mode)
    try:
        if path.suffix.lower()=='.csv':
            with path.open(newline='',encoding='utf-8-sig') as f:
                rd=csv.reader(f)
                for r in list(rd)[1:]:add(r)
        else:
            wb=load_workbook(path,read_only=True,data_only=True)
            for r in wb[sheet].iter_rows(min_row=2,values_only=True):add(r)
            wb.close()
    except Exception as exc:
        print('CANNOT SCAN:',path.name,type(exc).__name__);continue
    if len(dat)==628 and pins-set(dat)=={'600028'}:
        if results is not None and results[1]!=dat:die('Multiple raster sources disagree')
        results=((path,sheet,headers[xcol],headers[mcol]),dat)

if results:
    info,dat=results
    detailed=set(v[0] for v in dat.values())-{10}
    levels=set(1 if x in (11,12,13) else 2 if x in (21,22,23) else 3 for x in detailed)
    agreeable=sum(x==m for x,m in dat.values())
    if len(detailed)!=7 or len(levels)!=3:die('Expected all seven detailed classes and all three broad settlement classes')
    replacements={
        'Detailed non-water GHS-SMOD classes represented':f'{len(detailed)}/7',
        'Degree of Urbanisation Level-1 classes represented':f'{len(levels)}/3',
        'Exact-cell versus surrounding 3 × 3 GHS-SMOD modal agreement':f'{100*agreeable/628:.2f}%'
    }
    print('\nRASTER RECOMPUTATION | VERIFIED PIN-LEVEL SOURCE')
    print('Source:',str(info[0].relative_to(DEM)),'sheet:',info[1])
    print('Exact-cell field:',info[2],'| 3x3 mode field:',info[3])
    print('Detailed classes represented:',sorted(detailed),f'({len(detailed)}/7)')
    print('Level-1 classes represented:',sorted(levels),f'({len(levels)}/3)')
    print('Exact/modal agreement:',agreeable,'/',628,f'= {100*agreeable/628:.2f}%')
else:
    replacements={}
    print('\nRASTER RECOMPUTATION | SOURCE NOT YET IDENTIFIED')
    print('The three raster diagnostics cannot be certified from an aggregated four-class CSV.')

# Create scientific-style candidate from original 26 rows, without altering originals.
bylabel['Unique valid-format PIN codes']['Revised result']=str(unique)
bylabel['Unique valid-format PIN codes']['Interpretation']='Distinct six-digit PIN codes before spatial matching'
bylabel['Current Tamil Nadu districts represented']['Diagnostic']='Districts represented'
bylabel['Current Tamil Nadu districts represented']['Revised result']='37/38 (97.4%)'
bylabel['Current Tamil Nadu districts represented']['Interpretation']='Districts with at least one matched postal reference location'
bylabel['Current district not represented']['Diagnostic']='District not represented'
bylabel['Current district not represented']['Revised result']='Chengalpattu'
bylabel['Current district not represented']['Interpretation']='No matched household PIN locations in this district'
for row in s31:
    if row['Diagnostic'] in replacements:
        row['Revised result']=replacements[row['Diagnostic']]
        if row['Diagnostic']=='Detailed non-water GHS-SMOD classes represented':
            row['Interpretation']='Breadth of detailed settlement environments sampled'
        elif row['Diagnostic']=='Degree of Urbanisation Level-1 classes represented':
            row['Interpretation']='Rural areas, towns and semi-dense areas, and cities'
        else:
            row['Interpretation']='Proportion of PIN locations with matching exact-cell and local modal classes'

complete=all('PENDING' not in r['Revised result'] and 'confirm' not in r['Revised result'].lower() for r in s31)
output=OUT/('TABLE_S31_N6039_PUBLICATION_READY.csv' if complete else 'TABLE_S31_N6039_WORKING_NOT_FOR_PUBLICATION.csv')
save_csv(output,['Diagnostic','Result','Interpretation'],[[r['Diagnostic'],r['Revised result'],r['Interpretation']] for r in s31])
report={'N':6039,'normalized_unique_PINs':unique,'missing_district':'Chengalpattu','complete':complete,
        'raster_source':str(results[0][0]) if results else None,
        'outstanding':[r['Diagnostic'] for r in s31 if 'PENDING' in r['Revised result']]}
(OUT/'STAGE7_TABLE_VALIDATION.json').write_text(json.dumps(report,indent=2))
print('\nPUBLICATION TABLE STATUS:', 'ALL 26 ROWS RESOLVED' if complete else '26 ROWS PRESERVED; THREE RASTER ROWS NEED SOURCE')
print('SAVED:',output)
print('Stage6 tables and figures untouched; Figure 1 and public archives untouched.')
print('='*78)
