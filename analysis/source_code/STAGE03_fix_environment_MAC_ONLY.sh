#!/usr/bin/env bash
# WSQS N6039: isolated scientific Python environment; no model fits, no dataset edits.
set -Eeuo pipefail
umask 077

ROOT="$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_WSQS_RERUN_20261030_N6039"
CODE="$ROOT/02_code_private/reconciled/WSQS_Study_Repro_N6039"
LOCK="$ROOT/07_audit/COHORT_DATASET_LOCK_N6039.json"
PREFLIGHT="$ROOT/04_numerical_results/measurement/CORE8_AND_FEATURE_PREFLIGHT_N6039.json"
VENV="$ROOT/.venv_wsqs_n6039"
AUDIT="$ROOT/07_audit/SCIENTIFIC_ENVIRONMENT_N6039.json"

for f in "$CODE/environment/requirements-demo.txt" "$CODE/config/models.json" "$LOCK" "$PREFLIGHT"; do
  if [[ ! -f "$f" ]]; then echo "FAIL | Missing required file: $f" >&2; exit 1; fi
done

if [[ ! -x "$VENV/bin/python" ]]; then
  echo 'CREATE | Isolated Python environment (Anaconda base will not be modified)'
  python3 - <<'PY'
import sys
assert sys.version_info[:2] == (3, 11), f'Python 3.11 is required; found {sys.version.split()[0]}'
print('PASS | Python 3.11 available:', sys.executable)
PY
  python3 -m venv "$VENV"
fi

PY="$VENV/bin/python"
echo 'INSTALL | Pinned scientific libraries from original October 19 package'
PYTHONNOUSERSITE=1 "$PY" -m pip install --disable-pip-version-check --only-binary=:all: \
  -r "$CODE/environment/requirements-demo.txt"

export WSQS_ROOT="$ROOT"
export WSQS_CODE="$CODE"
export WSQS_VENV="$VENV"
export WSQS_AUDIT="$AUDIT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE/src"


printf '\nTEST | Existing scientific unit tests and synthetic demonstrations\n'
(
  cd "$CODE"
  "$PY" -m unittest discover -s tests -v
  "$PY" run.py --mode demo
  "$PY" run_extended.py --mode demo
)

"$PY" - <<'PY'
import hashlib
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import catboost
from wsqs.models import model_settings, partition_indices
from wsqs.features import construct_features
from wsqs.extended import scientific_plans

root = Path(os.environ['WSQS_ROOT'])
code = Path(os.environ['WSQS_CODE'])
venv = Path(os.environ['WSQS_VENV'])
audit = Path(os.environ['WSQS_AUDIT'])
assert Path(sys.prefix).resolve() == venv.resolve(), 'Not using the isolated venv'
expected = {
    'numpy': '1.26.4',
    'pandas': '2.3.3',
    'scipy': '1.13.1',
    'scikit-learn': '1.7.2',
    'catboost': '1.2.10',
}
observed = {name: md.version(name) for name in expected}
assert observed == expected, f'Pinned version mismatch: {observed}'
assert not any('anaconda3/lib/python' in str(getattr(m, '__file__', '')) for m in (np, pd, scipy, sklearn, catboost)), 'Imported package from Anaconda base'
fit, val, test, dev = partition_indices(6039, int(model_settings()['primary_seed']))
assert (len(fit), len(val), len(test), len(dev)) == (3864, 967, 1208, 4831)
assert len(scientific_plans()['full']) == 33
lock = json.loads((root/'07_audit/COHORT_DATASET_LOCK_N6039.json').read_text())
preflight = json.loads((root/'04_numerical_results/measurement/CORE8_AND_FEATURE_PREFLIGHT_N6039.json').read_text())
assert preflight['status'] == 'SCORING_AND_FEATURE_PREFLIGHT_PASSED'
assert preflight['source_sha256'] == lock['output_sha256']['primary'], 'Cohort/preflight hash mismatch'
req = code/'environment/requirements-demo.txt'
report = {
    'status': 'ISOLATED_SCIENTIFIC_ENVIRONMENT_VERIFIED',
    'purpose': 'WSQS N6039 scientific modeling; no empirical models fitted by this setup',
    'python': sys.version.split()[0],
    'python_executable': sys.executable,
    'venv': str(venv),
    'versions': observed,
    'requirements_demo_sha256': hashlib.sha256(req.read_bytes()).hexdigest(),
    'source_cohort_sha256': preflight['source_sha256'],
    'scientific_preflight_status': preflight['status'],
    'partition': {'fitting': len(fit), 'validation': len(val), 'heldout': len(test)},
    'scientific_feature_count': 33,
}
if audit.exists():
    previous = json.loads(audit.read_text())
    assert previous['source_cohort_sha256'] == report['source_cohort_sha256'], 'Existing audit references a different cohort'
    print('PASS | Existing environment audit has matching cohort (no overwrite)')
else:
    audit.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS | Environment verification audit saved:', audit)
print('PASS | Isolated Python and exact pinned libraries:', observed)
print('PASS | Preflight cohort SHA-256 matches frozen lock')
print('PASS | 33 predictors and 3864/967/1208 split configuration')
PY

printf '\n===============================================================\n'
printf 'PASS | Isolated WSQS N6039 environment verified\n'
printf 'VENV_PYTHON: %s\n' "$PY"
printf 'AUDIT: %s\n' "$AUDIT"
printf 'NO empirical model fitted; existing preflight and frozen data untouched.\n'
printf '===============================================================\n'
