#!/usr/bin/env python3
"""Stage 12: verified, privacy-filtered N6039 numerical handoff for exhibit assembly.

Reads frozen Stage11 report and per-stage files; does not fit models, modify upstream
files, or copy respondent-level data. Deliberately DOES NOT include descriptive
sidecar data, raw survey CSVs, model checkpoints or prior figure artwork.

Output: ~/Desktop/WSQS_N6039_VERIFIED_AGGREGATES_STAGE12.zip

Run with the N6039 scientific venv or any Python >=3.9; only stdlib is needed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

PROJECT = 'NPJ_WSQS_RERUN_20261030_N6039'
SOURCE_SHA = 'c72b8480d3cff223d1d4cf0ebcc29a3dd80204d563cd63a4295f6739f0ed77a1'
STAGE_NAMES = {
    '04':'STAGE04_PRIMARY_CROSSDOMAIN_N6039',
    '05':'STAGE05_SENSITIVITY_SATISFACTION_N6039',
    '06':'STAGE06_REPEATED_SPLITS_BENCHMARK_N6039',
    '07':'STAGE07_PRIORITY_NULL_B1000_N6039',
    '08':'STAGE08_SHAP_ATTRIBUTION_N6039',
    '09':'STAGE09_SERVICE_RESPONSE_B200_N6039',
    '10':'STAGE10_MEASUREMENT_PROVIDER_N6039',
}
AUDIT_FILES = (
    '01_verified_source_file_ledger.csv',
    '02_exhibit_56_dependency_register.csv',
    '03_numerical_audit_summary.json',
    '04_handoff_and_pending_exhibits.md',
)
DISALLOWED_HEADER = {
    'responseid','respondentid','participantid','surveyid','recordid',
    'ipaddress','emailaddress','phonenumber','firstname','lastname',
}


def require(test, msg):
    if not test:
        raise RuntimeError(msg)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text('utf-8'))


def parse_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def check_source_header(path):
    if path.suffix.lower()!='.csv':
        return
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        headers=next(csv.reader(f),[])
    normalized={''.join(c for c in str(h).lower() if c.isalnum()) for h in headers}
    bad=DISALLOWED_HEADER & normalized
    require(not bad, 'Potential respondent identifier in aggregate-file header '+str(path)+': '+str(sorted(bad)))


def validate_and_prepare(work):
    work=Path(work).expanduser().resolve()
    source=work/'00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    require(source.is_file() and sha256(source)==SOURCE_SHA,
            'Frozen cohort checksum failed. Refusing to export.')
    ad=work/'07_audit/STAGE11_NUMERICAL_EXHIBIT_AUDIT_N6039'
    for f in AUDIT_FILES:
        require((ad/f).is_file(), 'Missing Stage11 audit file '+f)
    audit=load_json(ad/AUDIT_FILES[2])
    require(audit.get('status')=='PASS' and audit.get('source_sha256')==SOURCE_SHA,
            'Stage11 PASS/source checksum not confirmed')
    require(int(audit.get('verified_numeric_result_files',-1))==49,
            'Stage11 expected exactly 49 numerically verified output files')
    require(audit.get('counts_by_exhibit_status')=={
        'NUMERIC_SOURCE_PRESENT':36,
        'AWAITING_SIDECAR':10,
        'STATIC_METHOD_REVIEW':10,
    }, 'Stage11 status count differs from recorded handoff')
    for number, name in STAGE_NAMES.items():
        p=work/'07_audit'/(name+'_COMPLETE.json')
        expected=audit['stage_audit_sha256'][number]
        require(p.is_file() and sha256(p)==expected,
                f'Stage {number} completed-audit checksum mismatch')
    ex=parse_csv(ad/AUDIT_FILES[1])
    require(len(ex)==56 and len({r['exhibit'] for r in ex})==56,
            '56-exhibit register malformed')
    ledger=parse_csv(ad/AUDIT_FILES[0])
    require(len(ledger)==49, 'Stage11 verified source ledger count changed')
    require(len({r['relative_path'] for r in ledger})==49,'Duplicate source path')
    chosen=[]
    excluded=[]
    for row in ledger:
        rel=row['relative_path']
        # Never allow absolute/traversal or paths outside the known numeric source tree.
        relpath=Path(rel)
        require(not relpath.is_absolute() and '..' not in relpath.parts
                and rel.startswith('04_numerical_results/'),
                'Unrecognized/non-numeric source path: '+rel)
        require(relpath.suffix.lower() in {'.csv','.json'},'Unexpected source extension '+rel)
        path=work/relpath
        require(path.is_file(), 'Missing scientific source '+rel)
        require(sha256(path)==row['sha256'], 'Scientific-source SHA mismatch: '+rel)
        require(path.stat().st_size==int(row['bytes']), 'Scientific-source byte count changed: '+rel)
        # Explicit private flag plus filename-based safeguard.
        is_private=row['privacy'].strip().upper()!='AGGREGATE' or 'PRIVATE' in path.name.upper()
        if is_private:
            excluded.append(rel)
            continue
        check_source_header(path)
        chosen.append((rel,path,row['sha256'],int(row['bytes'])))
    require(len(chosen)==47 and len(excluded)==2,
            f'Expected 47 aggregate and 2 private exclusions, found {len(chosen)} and {len(excluded)}')
    return work,ad,chosen,excluded,audit


def build_package(work, output, check_only=False):
    work,ad,chosen,excluded,audit=validate_and_prepare(work)
    print('='*75)
    print('STAGE12 | VERIFIABLE AGGREGATE HANDOFF — NO RESPONDENT ROWS')
    print('PASS | Source cohort and seven Stage04–10 audit hashes verified')
    print(f'PASS | All {len(chosen)+len(excluded)} Stage11 scientific files re-hashed')
    print(f'PASS | {len(chosen)} aggregate files eligible; {len(excluded)} PRIVATE files excluded')
    print('PASS | Stage11 56-exhibit dependency register available')
    if check_only:
        print('CHECK ONLY | No archive created')
        print('='*75)
        return
    out=Path(output).expanduser().resolve()
    require(out!=work and not out.exists(), 'Output archive already exists or invalid: '+str(out))
    out.parent.mkdir(parents=True,exist_ok=True)
    metadata={
        'purpose':'PRIVATE COLLABORATIVE EDITORIAL HANDOFF ONLY; not a public reproducibility package',
        'status':'STAGE11 VERIFIED NUMERICAL SOURCES; FIGURES NOT RENDERED',
        'source_sha256':SOURCE_SHA,
        'stage11_exhibit_counts':audit['counts_by_exhibit_status'],
        'scientific_files_verified':49,
        'aggregate_files_included':len(chosen),
        'private_files_excluded_count':len(excluded),
        'private_files_excluded':[Path(p).name for p in excluded],
        'not_included':['respondent-level survey records','private SHAP rows',
                        'private household weighting scores','fitted models/checkpoints',
                        'separate descriptive sidecar','historical SVG layouts'],
        'included_sources':[{'path':r, 'sha256':h,'bytes':size} for r,_,h,size in chosen],
        'cautions':['36 exhibits numerically sourced, not typeset or rendered',
                    '10 original exhibits await independent descriptive sidecar',
                    '10 original exhibits require static-method review',
                    'figure and table text must be reconciled against full-precision sources',
                    'review rare-cell disclosure before any public release'],
    }
    readme=(
        'WSQS N6039 — stage12 VERIFIED AGGREGATE HANDOFF\n\n'
        'This archive contains 47 hash-verified aggregate numerical source files from\n'
        'Stages 04–10 plus the Stage11 56-exhibit register and numerical audit.\n'
        'It excludes both explicitly PRIVATE numerical files and ALL respondent-level\n'
        'data, model checkpoints, original/updated figure artwork and the descriptive\n'
        'sidecar. It is not a public-release research data package.\n\n'
        'Stage11 status: 36/56 original exhibits have numeric sources; 10 still need\n'
        'the independent geographic/temporal/demographic sidecar; 10 need a static\n'
        'method/definition review. None are certified as rendered manuscript exhibits.\n\n'
        'To resume: use stage11/02_exhibit_56_dependency_register.csv to map\n'
        'the numeric_sources/ content to figures/tables; later merge separately\n'
        'verified sidecar sources and locked historic figure layouts.\n'
    )
    handle,tmp_name=tempfile.mkstemp(prefix='.WSQS_STAGE12_',suffix='.tmp',dir=str(out.parent))
    os.close(handle)
    temp=Path(tmp_name)
    try:
        with ZipFile(temp, 'w', compression=ZIP_DEFLATED, compresslevel=7, allowZip64=True) as z:
            for rel,path,_,_ in chosen:
                z.write(path,arcname='numeric_sources/'+rel)
            for f in AUDIT_FILES:
                z.write(ad/f,arcname='stage11/'+f)
            z.writestr('HANDOFF_README.txt', readme)
            z.writestr('handoff_manifest.json',json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        with ZipFile(temp,'r') as z:
            require(z.testzip() is None, 'Zip CRC failure')
            require(len(z.namelist())==53,'Unexpected package member count')
            require(len(set(z.namelist()))==len(z.namelist()),'Duplicate archive members')
            for rel,_,digest,_ in chosen:
                require(hashlib.sha256(z.read('numeric_sources/'+rel)).hexdigest()==digest,
                        'Archive internal source hash mismatch: '+rel)
        require(not out.exists(),'Refusing to overwrite existing handoff')
        os.replace(temp,out)
    finally:
        if temp.exists():temp.unlink()
    print('PASS | ZIP CRC, manifest and all 47 embedded scientific file SHA-256 values')
    print('ZIP:',out)
    print('BYTES:',out.stat().st_size)
    print('NEXT | Upload the single ZIP for exhibit-source inspection and locked-layout reconstruction')
    print('='*75)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path.home()/
                   'Desktop/Spring 2026/Chapter 3 MANUAL RERUN'/PROJECT)
    p.add_argument('--output',type=Path,
                   default=Path.home()/'Desktop/WSQS_N6039_VERIFIED_AGGREGATES_STAGE12.zip')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    build_package(a.work,a.output,a.check_only)

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print('FAILED |',type(e).__name__,str(e),file=sys.stderr,flush=True)
        raise
