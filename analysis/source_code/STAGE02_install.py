#!/usr/bin/env python3
"""WSQS N6039 | Stage 02, import original study reproducibility package.

Does NOT modify historical files, publish files, train any model, or overwrite
existing outputs. Use after the locked cohort builder has passed on the Mac.

From downloaded and extracted stage bundle:
   python3 ~/Downloads/WSQS_STAGE02_REPRO_N6039/install.py

Optional: --base '/different/Chapter 3 MANUAL RERUN'
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

NAME = 'NPJ_WSQS_RERUN_20261030_N6039'
PUBLIC_ZIP = 'WSQS_Paper_Reproducibility_v0.1.0_REVIEW_20261019_051028.zip'
PUBLIC_ZIP_SHA256 = 'bf01b8c88da36b77ccf8112a9e644c1cf077c5b2c9caf8bbe79a8ddfb4fbccac'
FROZEN_OLD_ORIGINAL_ZIP_SHA = 'f94a1ed639dfb42706c8fff51b0071ff580a4cd6d42638b4e65177d458bb4ba1'
EXPECTED_OLD_MASTER_SHA = '9b1224ffd80397cf919c48cd944b7e7e3e17c2dded8dd99cab77ef4894b7ce9a'
BUNDLE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as stream:
        for buf in iter(lambda:stream.read(1<<20), b''):h.update(buf)
    return h.hexdigest()


def write_new(path, content):
    require(not path.exists(), 'Refusing to overwrite: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def immutable_copy(source, dest):
    if dest.is_file():
        require(sha(source)==sha(dest), 'Existing archived package differs: '+str(dest))
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        require(sha(source)==sha(dest), 'Archive copy digest mismatch')
    return sha(dest)


def read_csv_id_audit(path):
    rows = 0; ids = set(); order_h = hashlib.sha256()
    with path.open('r',encoding='utf-8-sig',newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        names = {''.join(c for c in x.lower() if c.isalnum()):x for x in fields}
        col = names.get('responseid')
        require(col is not None,'Missing ResponseId in primary cohort')
        require(len(fields)==182, 'Expected preserved 182 columns, observed '+str(len(fields)))
        for row in reader:
            rid = row[col].strip(); require(rid and rid not in ids,'Duplicate or blank ResponseId')
            ids.add(rid); order_h.update((rid+'\n').encode());rows+=1
    require(rows==6039, f'Primary cohort expected N6039, got N{rows}')
    return order_h.hexdigest()


# The ONLY source modifications needed to make the portable historical package
# executable on the locked primary 6039 cohort. Model settings, seeds, split
# algorithm, feature definitions and estimator methods remain unchanged.
PATCHES = {
    'src/wsqs/models.py': [
        ('3932, 984, 1229', '3864, 967, 1208'),
        ('"Reproduction requires the 6,145-household dataset"', '"Reanalysis requires the locked 6,039-household primary cohort"'),
        ('if n != 6145 or X.shape[1] != 33 or len(y) != n:', 'if n != 6039 or X.shape[1] != 33 or len(y) != n:'),
        ('"Expected the study\'s 6,145 × 33 analytical inputs"', '"Expected the locked 6,039 × 33 analytical inputs"'),
    ],
    'src/wsqs/extended.py': [
        ('len(frame) != 6145', 'len(frame) != 6039'),
        ('X.shape != (6145, 33)', 'X.shape != (6039, 33)'),
    ],
    'run.py': [
        ('len(frame) != 6145', 'len(frame) != 6039'),
    ],
    'tests/test_models.py': [
        ('6145, 20260714', '6039, 20260714'),
        ('(3932, 984, 1229, 4916)', '(3864, 967, 1208, 4831)'),
        ('            6145,', '            6039,'),
    ],
}


def stage_copy(root, package_zip, destination):
    with ZipFile(package_zip) as z:
        require(z.testzip() is None,'Damaged source ZIP archive')
        names = {i.filename for i in z.infolist() if not i.is_dir()}
        require(len(names)==54,'Historical public ZIP content changed; expected 54 files, got '+str(len(names)))
        for name in names:
            path = Path(name)
            require(not path.is_absolute() and '..' not in path.parts, 'Unsafe archive member: '+name)
        required = set(PATCHES)|{'config/models.json','config/feature_spec.json','config/scientific_plans.json','run_extended.py'}
        require(required.issubset(names),'Archive missing required sources: '+str(required-names))
        destination.mkdir(parents=True,exist_ok=False)
        for item in z.infolist():
            if item.is_dir():continue
            name = item.filename
            # Historical public aggregate numbers belong ONLY to the protected reference archive.
            if name.startswith('results/aggregate/'):
                continue
            path = destination/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(z.read(name))

    records=[]
    for filename,replacements in PATCHES.items():
        p=destination/filename
        source=p.read_text(encoding='utf-8')
        s=source
        for before,after in replacements:
            require(s.count(before)==1,filename+' expected exactly one anchor: '+repr(before))
            s=s.replace(before,after,1)
            records.append(dict(file=filename,old=before,new=after))
        require(s!=source,'No changes in '+filename)
        p.write_text(s,encoding='utf-8')
    # Preserve the original README but prepend a prominently identified run-specific note.
    user_note='''# N6039 Working Copy (PRIVATE — not an approved public release)\n\nThis is an N=6039 adaptation of the October 19 v0.1.0 study-specific\nreproducibility archive. The original archive and scientific specifications\nare preserved separately. Historical performance and counts below belong\nto N=6145; do not cite them as results for the new cohort.\n\nMinimal patches: only old cohort-count assertions and partition-size tests.\nThe split algorithm, seeds, scoring and predictor definitions are unchanged.\nThe original archive's old empirical aggregate CSVs are deliberately NOT\nincluded in this active working copy. A separate historical archive preserves them.\n\nNew analyses and complete figure/table sources MUST be implemented and checked\nseparately. Do not claim that this portable starter package covers every\nscientific exhibit. The private code collection is NOT part of this folder.\n\n---\n\n'''
    p=destination/'README.md'; p.write_text(user_note+p.read_text(encoding='utf-8'),encoding='utf-8')
    (destination/'N6039_MODIFICATIONS.json').write_text(json.dumps({'source_zip_sha256':PUBLIC_ZIP_SHA256,'cohort':6039,'patches':records,'unchanged': ['config/models.json','config/feature_spec.json','config/scientific_plans.json']},indent=2)+'\n',encoding='utf-8')
    for file in ('run.py','run_extended.py','src/wsqs/models.py','src/wsqs/extended.py','tests/test_models.py'):
        py_compile.compile(str(destination/file),doraise=True)
    return records


def main():
    os.umask(0o077)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,default=Path.home()/'Desktop/Spring 2026/Chapter 3 MANUAL RERUN')
    ap.add_argument('--skip-tests',action='store_true',help='Only for missing Python dependencies; not recommended')
    args=ap.parse_args()
    project=args.base.expanduser().resolve()/NAME
    cohort=project/'00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    lock=project/'07_audit/COHORT_DATASET_LOCK_N6039.json'
    source=BUNDLE/PUBLIC_ZIP
    require(source.is_file() and sha(source)==PUBLIC_ZIP_SHA256, 'Bundled original reproducibility archive missing/changed')
    require(cohort.is_file() and lock.is_file(), 'Run the cohort-construction script successfully first')
    old=json.loads(lock.read_text(encoding='utf-8'))
    require(old.get('status')=='COHORT_DATASETS_CONSTRUCTED_AND_VERIFIED','Cohort freeze status unexpected')
    require(old.get('primary_n')==6039 and old.get('master_sha256')==EXPECTED_OLD_MASTER_SHA,'Cohort master/source lock disagrees')
    expected_hash=old.get('output_sha256',{}).get('primary')
    require(expected_hash==sha(cohort),'Frozen primary dataset SHA-256 does not match lock audit')
    ordered_hash=read_csv_id_audit(cohort)
    require(ordered_hash==old.get('primary_ordered_response_ids_sha256'),'Ordered cohort ResponseIds differ from frozen lock')
    print('PASS | Frozen N6039 primary cohort integrity verified')
    print('PASS | Original reproducibility ZIP identity verified')
    archives=project/'02_code_private/archive'
    archives.mkdir(parents=True,exist_ok=True)
    original=archives/PUBLIC_ZIP
    immutable_copy(source,original)
    active=project/'02_code_private/reconciled/WSQS_Study_Repro_N6039'
    prior_audit=project/'07_audit/REPRO_SOURCE_INTEGRATION_N6039.json'
    if active.exists():
        require(prior_audit.is_file(),'Working copy exists but integration audit missing; inspect manually')
        a=json.loads(prior_audit.read_text(encoding='utf-8'))
        require(a['source_zip_sha256']==PUBLIC_ZIP_SHA256 and a['input_sha256']==expected_hash,'Prior integration audit differs')
        for key,value in a['working_code_sha256'].items():
            require((active/key).is_file() and sha(active/key)==value,'Working source changed: '+key)
        print('PASS | Previous integration verified; no files overwritten')
    else:
        # Write into a temporary sibling, and install atomically after code checks.
        active.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='.stage02_',dir=str(active.parent)))
        try:
            records=stage_copy(project, original, stage/'source')
            shutil.move(str(stage/'source'),str(active))
        finally:
            if stage.exists():shutil.rmtree(stage)
        code_files=list(PATCHES)+['config/models.json','config/feature_spec.json','config/scientific_plans.json']
        audit={
            'status':'ORIGINAL_REPRO_SOURCE_INTEGRATED_N6039',
            'date_utc':datetime.now(timezone.utc).isoformat(),
            'source_zip_sha256':PUBLIC_ZIP_SHA256,
            'historical_frozen_original_zip_sha256':FROZEN_OLD_ORIGINAL_ZIP_SHA,
            'input_sha256':expected_hash,
            'ordered_response_ids_sha256':ordered_hash,
            'working_directory':str(active),
            'working_code_sha256':{f:sha(active/f) for f in code_files},
            'patch_count':len(records),
            'model_specs_changed':False,
            'models_fit':0,
            'historical_aggregate_values_imported_to_active':False,
            'public_release_authorized':False,
            'scope':'Portable reproduction foundation, not complete manuscript analysis',
        }
        write_new(prior_audit,json.dumps(audit,indent=2)+'\n')
        print('PASS | Reproducibility package incorporated into protected N6039 workspace')
        print('PASS | Strict, minimal cohort adaptation of 4 code/test files')
        print('PASS | Original model configs and feature schema untouched')
        print('PASS | Historical aggregate results excluded from active analytical source')

    # Stage the new public release as a manifest only, pending complete empirical run.
    target=project/'06_public_reproducibility/RELEASE_PREPARATION_STATUS.json'
    if not target.exists():
        write_new(target,json.dumps({
            'status':'NOT_PUBLISHED',
            'based_on':'WSQS original-study reproducibility v0.1.0',
            'source_location':'02_code_private/reconciled/WSQS_Study_Repro_N6039',
            'target_n':6039,
            'must_add':['priority-null permutation','8 C+P cross-domain baselines','model-derived SHAP and intervals','service-response B200 refits','CORE8 measurement diagnostics','Q6.1 comparison'],
            'private_respondent_records_in_public_repo':False,
            'published':False,
        },indent=2)+'\n')

    if not args.skip_tests:
        for label,command in [
            ('Unit tests',[sys.executable,'-m','unittest','discover','-s','tests','-v']),
            ('Synthetic benchmark demo',[sys.executable,'run.py','--mode','demo']),
            ('Synthetic scientific demo',[sys.executable,'run_extended.py','--mode','demo']),
        ]:
            env=os.environ.copy();env['PYTHONPATH']=str(active/'src')
            r=subprocess.run(command,cwd=active,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            if r.returncode:
                print(r.stdout[-5000:]);raise RuntimeError(label+' failed; see printed output')
            print('PASS | '+label)
    else:
        print('WARNING | Python smoke tests explicitly skipped; scientific models must not start until tests pass')

    print('SOURCE:',active)
    print('ORIGINAL:',original)
    print('DATA:',cohort)
    print('AUDIT:',prior_audit)
    print('PASS | Stage 02 complete. No empirical model fitted.')

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError) as e:
        print('FAILED |',e,file=sys.stderr);sys.exit(1)
