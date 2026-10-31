#!/usr/bin/env python3
"""WSQS Stage14 | Merge the 47 verified Stage12 aggregates and eight Stage13
figure-source aggregates into one integrity-checked, privacy-conscious handoff.

Also records an inventory (names/sizes only) of legacy main figure artwork in the
known old output directory; DOES NOT export any artwork or private household rows.
Read-only toward the WSQS research directories. No model fitting, no rewriting.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT_NAME = 'NPJ_WSQS_RERUN_20261030_N6039'
EXPECTED_COHORT_SHA = 'c72b8480d3cff223d1d4cf0ebcc29a3dd80204d563cd63a4295f6739f0ed77a1'
STAGE13 = 'STAGE13_PLOT_SOURCE_GAPS_N6039'
EXPECTED_13 = [
    '01_F2_priority_rating_heterogeneity.json',
    '02_F2d_individual_minus_equal_histogram.csv',
    '03_F3a_domain_pearson_correlations.csv',
    '04_F3b_equal_WSQS_quintile_domain_profiles.csv',
    '05_F3c_weighting_ecdf_grid.csv',
    '06_F4a_provision_mean_bootstrap_B750.csv',
    '07_TABLE03_p95_absolute_shift_and_correlations.csv',
    '08_TABLE05_net_absolute_block_SHAP.csv',
]
BAD_HEADERS = {'responseid','respondentid','participantid','surveyid',
               'recordid','ipaddress','emailaddress','phonenumber'}

def require(flag, why):
    if not flag: raise RuntimeError(why)

def sha_bytes(content): return hashlib.sha256(content).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def safe_csv_header(data, label):
    first=next(csv.reader(io.StringIO(data.decode('utf-8-sig'))),[])
    simplified={''.join(c.lower() for c in str(k) if c.isalnum()) for k in first}
    require(not BAD_HEADERS.intersection(simplified),
            f'Potential respondent identifiers in CSV header: {label}')


def check_stage12(stage12):
    require(stage12.is_file(),f'Missing Stage12 handoff ZIP: {stage12}')
    with ZipFile(stage12) as z:
        require(z.testzip() is None,'Stage12 ZIP CRC failure')
        allnames=z.namelist()
        require(len(set(allnames))==len(allnames),'Stage12 duplicated ZIP member')
        require('handoff_manifest.json' in allnames,'Missing Stage12 manifest')
        manifest=json.loads(z.read('handoff_manifest.json'))
        require(manifest.get('source_sha256')==EXPECTED_COHORT_SHA,
                'Stage12 source cohort digest mismatch')
        rows=manifest.get('included_sources',[])
        require(len(rows)==47 and manifest.get('private_files_excluded_count')==2,
                'Stage12 expected exactly 47 aggregate files and two private exclusions')
        expected={'numeric_sources/'+r['path'] for r in rows}
        require(all('..' not in PurePosixPath(n).parts and not n.startswith('/') for n in allnames),
                'Unsafe Stage12 ZIP path')
        require(set(n for n in allnames if n.startswith('numeric_sources/'))==expected,
                'Stage12 aggregate manifest does not match ZIP entries')
        copied={}
        for r in rows:
            n='numeric_sources/'+r['path']
            data=z.read(n)
            require(len(data)==int(r['bytes']) and sha_bytes(data)==r['sha256'],
                    f'Stage12 embedded SHA mismatch: {n}')
            if n.endswith('.csv'): safe_csv_header(data,n)
            copied[n]=data
        for n in allnames:
            if n.startswith('stage11/') or n in ('handoff_manifest.json','HANDOFF_README.txt'):
                copied[n]=z.read(n)
        require(len([n for n in copied if n.startswith('stage11/')])==4,
                'Expected four Stage11 audit files')
    return copied,manifest,sha(stage12)


def check_stage13(work):
    data=work/'00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    auditpath=work/'07_audit'/(STAGE13+'_COMPLETE.json')
    outdir=work/'04_numerical_results/figure_sources'/STAGE13
    require(data.is_file() and sha(data)==EXPECTED_COHORT_SHA,
            'Frozen private cohort not found or SHA mismatch')
    require(auditpath.is_file(),'Stage13 completion audit not found')
    a=read_json(auditpath)
    require(a.get('status')=='PASS','Stage13 audit not PASS')
    require(a.get('signature_inputs',{}).get('cohort')==EXPECTED_COHORT_SHA,
            'Stage13 audit references a different source cohort')
    require(a.get('models_fitted')==0 and a.get('respondent_ids_exported')==0 and
            a.get('heldout_rows_exported')==0,
            'Stage13 audit lacks required no-model/no-private-row assurances')
    require(a.get('outputs')==EXPECTED_13,
            'Stage13 eight source file list differs from locked expectation')
    hashes=a.get('output_sha256',{})
    require(set(hashes)==set(EXPECTED_13),'Stage13 audit output digest count differs')
    copied={}
    for n in EXPECTED_13:
        p=outdir/n
        require(p.is_file(),f'Missing Stage13 aggregate {n}')
        require(p.stat().st_size<2_000_000,'Stage13 aggregate unexpectedly large: '+n)
        content=p.read_bytes()
        require(sha_bytes(content)==hashes[n],'Stage13 source hash differs: '+n)
        if n.endswith('.csv'): safe_csv_header(content,n)
        copied['figure_sources/'+STAGE13+'/'+n]=content
    # Preserve verified source program as an additional reproducibility artifact.
    sourcep=Path.home()/'Downloads/WSQS_STAGE13_PLOT_SOURCE_GAPS_N6039.py'
    require(sourcep.is_file() and sha(sourcep)==a['signature_inputs']['script'],
            'Original Stage13 source is absent or digest differs from audit; do not guess provenance')
    copied['source_code/WSQS_STAGE13_PLOT_SOURCE_GAPS_N6039.py']=sourcep.read_bytes()
    copied['stage13/STAGE13_PLOT_SOURCE_GAPS_N6039_COMPLETE.json']=auditpath.read_bytes()
    return copied,a,sha(auditpath)


def layout_inventory(work):
    # Inventory only; these files are NOT bundled, since historical scatter plots
    # may contain point-level traces. Users review figure assets before sharing.
    directory=work.parent/'NPJ_Revision_2026/04_results/FIGURES_FOR_MANUSCRIPT_20261014_000258_892'
    payload={'directory':str(directory),'exists':directory.is_dir(),
             'artwork_included_in_handoff':False,'files':[]}
    if not directory.is_dir(): return payload
    listed=0
    for f in sorted(directory.rglob('*')):
        if not f.is_file() or f.is_symlink():continue
        if f.suffix.lower() not in ('.svg','.pdf','.png','.jpg','.jpeg','.tif','.tiff','.eps','.ai'):continue
        listed+=1
        if listed>1000: break
        payload['files'].append({'relative_path':f.relative_to(directory).as_posix(),
                                 'bytes':f.stat().st_size})
    payload['inventory_truncated']=listed>1000
    return payload


def create_archive(paths,out,manifest,stage12_hash,stage13_hash,layout):
    require(not out.exists(),f'Output exists, refusing to overwrite: {out}')
    require(len([x for x in paths if x.startswith('numeric_sources/')])==47,
            'Stage12 numeric count changed')
    require(len([x for x in paths if x.startswith('figure_sources/')])==8,
            'Stage13 figure-source count changed')
    meta={'status':'PASS','purpose':'AUTHOR-TO-EDITOR verified aggregate numerical handoff; not public respondent data',
          'source_cohort_sha256':EXPECTED_COHORT_SHA,
          'stage12_zip_sha256':stage12_hash,'stage13_audit_sha256':stage13_hash,
          'stage12_numeric_sources':47,'stage13_plot_sources':8,'aggregate_source_count':55,
          'historical_figure_artwork_included':False,'private_household_rows_included':False,
          'members':[{'name':k,'sha256':sha_bytes(v),'bytes':len(v)} for k,v in sorted(paths.items())],
          'caution':'Original figure SVGs, household scatter coordinates, descriptive sidecar, and model files are not in this archive.'}
    payload=dict(paths)
    payload['stage14/legacy_main_artwork_inventory_NAMES_ONLY.json']=(json.dumps(layout,indent=2)+'\n').encode()
    payload['stage14/verified_combined_handoff_manifest.json']=(json.dumps(meta,indent=2)+'\n').encode()
    payload['stage14/README.txt']=(
        'WSQS N6039 stage14: 47 original verified numeric sources plus eight Stage13 plotted aggregates.\n'
        'Stage11 56-exhibit mapping included. Private respondent rows/checkpoints and original figure\n'
        'artwork excluded. Legacy main-figure artwork inventory lists filenames only.\n'
        'Do not infer figure/table approval from numerical-source verification.\n').encode()
    out.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.wsqs_stage14_',suffix='.tmp',dir=out.parent)
    os.close(fd)
    p=Path(tmp)
    try:
        with ZipFile(p,'w',compression=ZIP_DEFLATED,compresslevel=7,allowZip64=True) as z:
            for n,b in sorted(payload.items()):z.writestr(n,b)
        with ZipFile(p,'r') as z:
            require(z.testzip() is None,'Final ZIP CRC failure')
            require(set(z.namelist())==set(payload),'ZIP entries differ from manifest')
            for n,b in payload.items():require(sha_bytes(z.read(n))==sha_bytes(b),f'Final ZIP hash mismatch: {n}')
        require(not out.exists(),'Refusing overwrite')
        os.replace(p,out)
    finally:
        if p.exists():p.unlink()
    return len(payload)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',type=Path,default=Path.home()/'Desktop/Spring 2026/Chapter 3 MANUAL RERUN'/ROOT_NAME)
    ap.add_argument('--stage12',type=Path,default=Path.home()/'Desktop/WSQS_N6039_VERIFIED_AGGREGATES_STAGE12.zip')
    ap.add_argument('--output',type=Path,default=Path.home()/'Desktop/WSQS_N6039_VERIFIED_FIGURE_SOURCES_STAGE14.zip')
    args=ap.parse_args()
    work=args.work.expanduser().resolve()
    copied12,oldmeta,digest12=check_stage12(args.stage12.expanduser().resolve())
    copied13,audit13,digest13=check_stage13(work)
    require(set(copied12).isdisjoint(copied13),'Stage12 and Stage13 filenames overlap')
    layout=layout_inventory(work)
    count=create_archive({**copied12,**copied13},args.output.expanduser().resolve(),oldmeta,digest12,digest13,layout)
    print('='*72)
    print('PASS | STAGE14 verified combined aggregate-source handoff')
    print('PASS | 47 Stage12 aggregate sources independently verified')
    print('PASS | Eight Stage13 aggregate sources independently verified')
    print('PASS | Frozen cohort and Stage13 source-code audit fingerprints verified')
    print('PASS | No respondent rows, private checkpoints or figure artwork included')
    print('LEGACY MAIN LAYOUT ASSETS INVENTORIED:',len(layout['files']),'(names only)')
    print('ZIP MEMBERS:',count)
    print('OUTPUT:',args.output.expanduser().resolve())
    print('BYTES:',args.output.expanduser().resolve().stat().st_size)
    print('NEXT | Upload this ZIP for locked-layout figure and table preparation')
    print('='*72)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('FAILED |',type(e).__name__,str(e),file=sys.stderr)
        raise
