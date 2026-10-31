#!/usr/bin/env python3
"""Finish WSQS Stage09 using ONLY the 200 already completed, verified refits.

This is a postfit recovery for an audit-metadata column typo in the initial
Stage09 runner. It does not alter the underlying Stage09 runner, frozen model,
scientific code, cohort or existing bootstrap checkpoints.

The original Stage09 code references eligibility['evaluation_n'] when writing
its analysis manifest, but summarize() produces the intentionally named
eligibility['eligible_heldout_households']. This wrapper corrects exactly that
one metadata lookup in memory. It also injects a checkpoint-existence guard
so NO model refitting can occur during recovery.

Run using the same isolated N6039 project Python:

  WORK="$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_WSQS_RERUN_20261030_N6039"
  "$WORK/.venv_wsqs_n6039/bin/python" \\
    ~/Downloads/WSQS_STAGE09_RECOVER_COMPLETED_B200_N6039.py

Written to the private stage_runners archive with a separate recovery note.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

PROJECT = 'NPJ_WSQS_RERUN_20261030_N6039'
STAGE = 'STAGE09_SERVICE_RESPONSE_B200_N6039'
ORIGINAL_FILENAME = 'WSQS_STAGE09_SERVICE_RESPONSE_B200_N6039.py'
# Exact SHA of the previously distributed original runner. Do not change it.
EXPECTED_SHA256 = 'df50ed765c37b4e3cff767a9ed06b1ce6bd3d910a9a0823f72251fb8f687f7ad'
COHORT_SHA256 = 'c72b8480d3cff223d1d4cf0ebcc29a3dd80204d563cd63a4295f6739f0ed77a1'
OLD_LOOKUP = "'profiles':{r['service_condition']:int(r['evaluation_n']) for r in eligibility.to_dict('records')}"
NEW_LOOKUP = "'profiles':{r['service_condition']:int(r['eligible_heldout_households']) for r in eligibility.to_dict('records')}"
REFIT_CHECK = "    if target_dir.exists():\n"
REFIT_GUARD = ("    check(target_dir.exists(), 'RECOVERY REFUSES TO FIT A MISSING B200 CHECKPOINT: '+str(target_dir))\n"
               + REFIT_CHECK)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for buf in iter(lambda:f.read(1048576),b''):
            h.update(buf)
    return h.hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def patch_source(original_path):
    raw = Path(original_path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256,
            'Original Stage09 source checksum differs from approved runner; REFUSING to patch')
    code = raw.decode('utf-8')
    require(code.count(OLD_LOOKUP) == 1, 'Metadata typo not found exactly once')
    require(code.count(REFIT_CHECK) == 1, 'Cannot uniquely install no-refit safety guard')
    code = code.replace(OLD_LOOKUP, NEW_LOOKUP, 1)
    code = code.replace(REFIT_CHECK, REFIT_GUARD, 1)
    require(code.count(OLD_LOOKUP) == 0 and code.count(NEW_LOOKUP)==1, 'Metadata patch failed')
    require(code.count('RECOVERY REFUSES TO FIT A MISSING B200 CHECKPOINT')==1,
            'No-refit guard missing')
    return compile(code, str(original_path), 'exec')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()

    source = Path.home() / 'Downloads' / ORIGINAL_FILENAME
    if args.self_test:
        # On a test machine, use the supplied original source if it is present.
        if not source.is_file() and Path('/mnt/data').is_dir():
            source = Path('/mnt/data') / ORIGINAL_FILENAME
        require(source.is_file(), 'Source unavailable for smoke test')
        patched = patch_source(source)
        require(patched is not None, 'Compiled patch unavailable')
        print('PASS | Exact original SHA and single-field metadata patch')
        print('PASS | Recovery-only checkpoint guard compiles')
        print('PASS | No dataset/model access and no model fitting')
        return

    os.umask(0o077)
    work = (Path.home()/'Desktop/Spring 2026/Chapter 3 MANUAL RERUN'/PROJECT).resolve()
    python = (work/'.venv_wsqs_n6039/bin/python').resolve()
    require(Path(sys.executable).resolve()==python,
            'Use locked isolated Python: '+str(python))
    require(source.is_file(), 'Missing unchanged original runner: '+str(source))
    compiled = patch_source(source)
    checkpoints = work/'03_run/checkpoints'/STAGE
    require(checkpoints.is_dir(), 'Missing Stage09 refit checkpoint directory')
    stage04_audit = work/'07_audit/STAGE04_PRIMARY_CROSSDOMAIN_N6039_COMPLETE.json'
    stage08_audit = work/'07_audit/STAGE08_SHAP_ATTRIBUTION_N6039_COMPLETE.json'
    cohort = work/'00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    for path in (stage04_audit,stage08_audit,cohort):
        require(path.is_file(),'Missing frozen prerequisite: '+str(path))
    require(digest(cohort)==COHORT_SHA256, 'Frozen N6039 source SHA changed')
    require(read_json(stage04_audit)['status']=='PASS', 'Stage04 audit not passed')
    require(read_json(stage08_audit)['status']=='PASS', 'Stage08 audit not passed')

    # Refuse recovery unless EVERY B200 record exists. In-memory guard below
    # makes the original model-fitting branch unreachable even if a record is
    # removed after this precheck.
    for i in range(1, 201):
        r = checkpoints / f'refit_{i:04d}'
        for name in ('refit_manifest.json', 'scenario_means.json',
                     'refit_model.cbm', 'development_sample_indices_PRIVATE.npy'):
            require((r/name).is_file(), f'Recovery cannot fit missing checkpoint {i}: {name}')
        m = read_json(r/'refit_manifest.json')
        require(m.get('status')=='PASS' and m.get('replicate')==i and m.get('tree_count')==500,
                f'Bootstrap checkpoint {i} has inconsistent metadata')

    require((work/'02_code_private/stage_runners'/ORIGINAL_FILENAME).is_file(),
            'Missing original archived Stage09 runner')
    require(digest(work/'02_code_private/stage_runners'/ORIGINAL_FILENAME)==EXPECTED_SHA256,
            'Archived Stage09 runner does not match original source')

    recovery_archive = work/'02_code_private/stage_runners'/Path(__file__).name
    if recovery_archive.exists():
        require(digest(recovery_archive)==digest(__file__), 'Recovery script archive differs')
    else:
        shutil.copy2(__file__, recovery_archive)
        require(digest(recovery_archive)==digest(__file__), 'Recovery code archive copy failed')

    print('='*78,flush=True)
    print('STAGE09 POSTFIT RECOVERY | NO MODEL REFITTING ALLOWED',flush=True)
    print('PASS | Original frozen Stage09 source SHA:', EXPECTED_SHA256,flush=True)
    print('PASS | All 200 existing bootstrap record directories located',flush=True)
    print('PATCH | Eligibility metadata key only: evaluation_n -> eligible_heldout_households',flush=True)
    print('PROTECT | No-refit guard installed in memory; original script unchanged',flush=True)
    print('='*78,flush=True)

    # The code's __file__ deliberately remains the original immutable runner:
    # the scientific signature and checkpoint verification therefore match
    # all 200 previously fitted models. The modification affects only final
    # audit metadata and recovery safety. No original file is rewritten.
    ns = {'__name__':'__main__', '__file__':str(source.resolve()),
          '__package__':None, '__cached__':None}
    prev_argv = sys.argv
    try:
        sys.argv = [str(source)]
        exec(compiled, ns)
    finally:
        sys.argv = prev_argv

    completion = work/'07_audit'/(STAGE+'_COMPLETE.json')
    require(completion.is_file(), 'Stage09 did not produce its expected completion audit')
    summary = read_json(completion)
    require(summary.get('status')=='PASS' and summary.get('refit_count')==200,
            'Stage09 completion audit missing/incorrect')
    resultdir = work/'04_numerical_results/inference'/STAGE
    for name, expected in summary['result_sha256'].items():
        f = resultdir/name
        require(f.is_file() and digest(f)==expected,'Aggregate output checksum mismatch: '+name)
    report = work/'07_audit/STAGE09_POSTFIT_RECOVERY_N6039.json'
    record = {
        'status':'PASS',
        'method':'in-memory exact one-key metadata correction; no scientific-code changes or model fitting',
        'original_stage09_runner_sha256':EXPECTED_SHA256,
        'recovery_runner_sha256':digest(__file__),
        'cohort_sha256':COHORT_SHA256,
        'stage09_signature':summary['signature'],
        'reused_refits':200,
        'new_models_fitted':0,
        'original_runner_modified':False,
        'corrected_field':'eligibility.eligible_heldout_households (metadata manifest only)',
        'stage09_completion_audit_sha256':digest(completion),
        'results':str(resultdir),
    }
    if report.exists():
        require(read_json(report)==record, 'An existing recovery audit differs')
    else:
        report.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('PASS | Postfit metadata recovery complete; Stage09 output checksums verified',flush=True)
    print('PASS | 200 earlier model refits reused; ZERO additional fits',flush=True)
    print('RECOVERY AUDIT:',report,flush=True)


if __name__=='__main__':
    main()
