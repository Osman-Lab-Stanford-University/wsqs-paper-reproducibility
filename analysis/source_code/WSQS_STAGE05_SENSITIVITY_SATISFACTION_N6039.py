#!/usr/bin/env python3
"""WSQS Stage 05 | Existing eight sensitivity fits + new Q6.1 diagnostics.

Run with the isolated N6039 environment after Stage 04 has completed:
  WORK="$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_WSQS_RERUN_20261030_N6039"
  "$WORK/.venv_wsqs_n6039/bin/python" \
    ~/Downloads/WSQS_STAGE05_SENSITIVITY_SATISFACTION_N6039.py

NO new predictor, outcome, model, random split, or parameter specification.
Original population weights estimated on development records only.
Original outcome models use S+C (25 predictors); four additional-feature
models use the original full S+C+P predictors (33 or 34 variables).
The Q6.1 analysis is *within-survey descriptive sensitivity*, not validation
against an independent criterion (Q6.1 is already a CORE8 item).

Exact model-fit routine, checkpoint verification, and full-precision CSV
conventions are imported from the verified Stage04 source script.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT = 'NPJ_WSQS_RERUN_20261030_N6039'
STAGE = 'STAGE05_SENSITIVITY_SATISFACTION_N6039'
STAGE04 = 'STAGE04_PRIMARY_CROSSDOMAIN_N6039'


def require(test, message):
    if not test:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def corrs(a, b):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    require(len(x) == len(y) and len(x) >= 3 and np.isfinite(x).all() and np.isfinite(y).all(),
            'Invalid correlation input')
    return {'pearson_r': float(pearsonr(x, y).statistic),
            'spearman_rho': float(spearmanr(x, y).statistic)}


def atomic_json(path, item):
    require(not path.exists(), 'Refusing to overwrite existing result: ' + str(path))
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(item, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def save_csv(path, dataframe):
    require(not path.exists(), 'Refusing to overwrite existing result: ' + str(path))
    temp = path.with_name(path.name + '.partial')
    dataframe.to_csv(temp, index=False, float_format='%.17g')
    os.replace(temp, path)


def run(base, stage04_script):
    os.umask(0o077)
    work = base.expanduser().resolve() / PROJECT
    venv = work / '.venv_wsqs_n6039/bin/python'
    require(Path(sys.executable).resolve() == venv.resolve(), 'Use isolated N6039 scientific Python: ' + str(venv))
    code = work / '02_code_private/reconciled/WSQS_Study_Repro_N6039'
    source = work / '00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    lock = work / '07_audit/COHORT_DATASET_LOCK_N6039.json'
    previous = work / '07_audit' / (STAGE04 + '_COMPLETE.json')
    preflight = work / '04_numerical_results/measurement/CORE8_AND_FEATURE_PREFLIGHT_N6039.json'
    model_checkpoint = work / '03_run/checkpoints' / STAGE
    result_dir = work / '04_numerical_results/modeling' / STAGE
    measure_dir = work / '04_numerical_results/measurement' / STAGE
    audit_file = work / '07_audit' / (STAGE + '_COMPLETE.json')
    for p in (source, lock, previous, preflight, stage04_script,
              code / 'src/wsqs/extended.py', code / 'src/wsqs/features.py',
              code / 'src/wsqs/models.py', code / 'config/scientific_plans.json'):
        require(p.is_file(), 'Missing required frozen input/script: ' + str(p))
    primary_sha = digest(source)
    frozen = load_json(lock)
    old = load_json(previous)
    require(primary_sha == frozen['output_sha256']['primary'] == old['fingerprint']['source_cohort_sha256'],
            'Dataset SHA mismatch against earlier locked stages')
    require(old['status'] == 'PASS' and old['total_empirical_catboost_fits'] == 23,
            'Stage04 must be finished before Stage05')
    require(old['fingerprint']['stage04_script_sha256'] == digest(stage04_script),
            'Stage04 source changed since models were fitted; do not silently mix algorithms')

    stage04_result = work / '04_numerical_results/modeling' / STAGE04 / '01_primary_block_heldout_metrics.csv'
    require(stage04_result.is_file() and digest(stage04_result) == old['output_sha256'][stage04_result.name],
            'Stage04 output metrics changed')

    stage04_spec = importlib.util.spec_from_file_location('wsqs_stage04_verified', str(stage04_script))
    module = importlib.util.module_from_spec(stage04_spec)
    stage04_spec.loader.exec_module(module)
    require(module.STAGE == STAGE04, 'Stage04 import mismatch')

    sys.path.insert(0, str(code / 'src'))
    from wsqs.features import construct_features
    from wsqs.models import partition_indices, model_settings
    from wsqs.extended import (DOMAINS, scientific_plans, alternative_outcomes,
                               supplementary_features, block_columns)
    settings, plans = model_settings(), scientific_plans()
    frame = pd.read_csv(source, dtype={'ResponseId': str}, low_memory=False)
    require(frame.shape == (6039, 182), 'Expected locked dataset 6039x182')
    require(frame['ResponseId'].notna().all() and frame['ResponseId'].is_unique,
            'ResponseId uniqueness/availability failed')
    if 'municipal_supply_applicable' not in frame:
        require('source09M_municipal_eligible' in frame, 'No provider-applicability evidence')
        frame = frame.rename(columns={'source09M_municipal_eligible': 'municipal_supply_applicable'})
    X, categorical, numeric = construct_features(frame)
    require(list(X.columns) == plans['full'] and X.shape == (6039, 33),
            'Changed feature specification')
    fit, val, test, dev = partition_indices(6039, int(settings['primary_seed']))
    require((len(fit), len(val), len(test), len(dev)) == (3864, 967, 1208, 4831),
            'Original deterministic partition was modified')
    outcomes, weights, fallback = alternative_outcomes(frame, dev)
    primary = outcomes['CORE8'].reset_index(drop=True)
    require(np.isclose(float(primary.mean()),
                       float(load_json(preflight)['core8_mean_full_precision']), atol=1e-11),
            'CORE8 score differs from frozen Stage03')

    inputs = {'stage05_script': digest(Path(__file__).resolve()),
              'stage04_script': digest(stage04_script),
              'stage04_signature': old['signature'],
              'dataset_sha256': primary_sha,
              'models_json': digest(code / 'config/models.json'),
              'feature_spec_json': digest(code / 'config/feature_spec.json'),
              'scientific_plans_json': digest(code / 'config/scientific_plans.json'),
              'features_py': digest(code / 'src/wsqs/features.py'),
              'models_py': digest(code / 'src/wsqs/models.py'),
              'extended_py': digest(code / 'src/wsqs/extended.py')}
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    print('=' * 76)
    print('STAGE 05 | N6039 OUTCOME + FEATURE SENSITIVITIES + Q6.1 DIAGNOSTICS')
    print('SOURCE SHA256:', primary_sha)
    print('RUN SIGNATURE:', signature)
    print('=' * 76, flush=True)
    if audit_file.exists():
        old05 = load_json(audit_file)
        require(old05['signature'] == signature, 'Existing Stage05 audit has different signature')
        for output, sha in old05['output_sha256'].items():
            path = work / output
            require(path.is_file() and digest(path) == sha, 'Invalid completed result ' + output)
        print('PASS | Stage05 already finished; all aggregate result SHA256 values verified')
        return

    for d in (model_checkpoint, result_dir, measure_dir):
        d.mkdir(parents=True, exist_ok=True)

    # A. Measurement and satisfaction diagnostics; does not fit models.
    scale = (5.0 - pd.to_numeric(frame['Acceptability_raw_severity'], errors='coerce')) * 25.0
    require(scale.notna().all() and scale.isin([0, 25, 50, 75, 100]).all(),
            'Q6.1 must be observed for all respondents and have valid ordinal scoring')
    core7a = outcomes['CORE7A']
    core7b = outcomes['CORE7B']
    c8c7a = corrs(primary, core7a)
    score_rows = []
    for name, alternative in outcomes.items():
        diff = alternative - primary
        metrics = corrs(primary, alternative)
        score_rows.append({'outcome': name, 'households': 6039,
                           'mean_score_100': float(alternative.mean()),
                           'sd_score_100': float(alternative.std(ddof=1)),
                           'mean_absolute_change_vs_CORE8_100': float(diff.abs().mean()),
                           'median_absolute_change_vs_CORE8_100': float(diff.abs().median()),
                           'max_absolute_change_vs_CORE8_100': float(diff.abs().max()),
                           'spearman_vs_CORE8': metrics['spearman_rho'],
                           'pearson_vs_CORE8': metrics['pearson_r']})
    satisfaction_rows = []
    for name in ('CORE8', 'CORE7A', 'CORE7B'):
        satisfaction_rows.append({'outcome': name, 'n': 6039,
                                  **corrs(scale, outcomes[name]),
                                  'Q6_1_direction': '100=completely satisfied; 0=completely dissatisfied',
                                  'is_external_criterion': False})
    domain_scores = pd.DataFrame({d: (5.0 - pd.to_numeric(frame[f'{d}_raw_severity'], errors='coerce')) * 25.0
                                  for d in DOMAINS})
    grouped_rows, profile_rows = [], []
    for sev in (1, 2, 3, 4, 5):
        selection = (pd.to_numeric(frame['Acceptability_raw_severity'], errors='coerce') == sev)
        n = int(selection.sum())
        require(n > 0, 'Missing Q6.1 category; cannot produce complete profiles')
        q = core7a.loc[selection]
        grouped_rows.append({'Q6_1_severity_1_best_5_worst': sev,
                             'Q6_1_satisfaction_score_100': (5-sev)*25,
                             'n': n,
                             'CORE7A_mean_100': float(q.mean()),
                             'CORE7A_sd_100': float(q.std(ddof=1)) if n > 1 else None,
                             'CORE7A_p10_100': float(q.quantile(.1)),
                             'CORE7A_median_100': float(q.median()),
                             'CORE7A_p90_100': float(q.quantile(.9)),
                             'CORE8_mean_100': float(primary.loc[selection].mean())})
        for domain in DOMAINS:
            values = domain_scores.loc[selection, domain].dropna()
            profile_rows.append({'Q6_1_severity': sev, 'domain': domain,
                                 'n_observed': int(len(values)),
                                 'domain_mean_100': float(values.mean()),
                                 'domain_sd_100': float(values.std(ddof=1)) if len(values) > 1 else None})

    # B. Eight PRE-EXISTING model sensitivities (not expanded hypotheses).
    variant_rows = []
    sc = block_columns('SC', plans)
    require(len(sc) == 25, 'The historical S+C outcome feature set has changed')
    for outcome in ('CORE7A', 'CORE7B', 'POPULATION_PRIORITY', 'INDIVIDUAL_PRIORITY'):
        target = outcomes[outcome].reset_index(drop=True)
        meta = module.train_and_record('OUTCOME_' + outcome, sc, target, fit, val, test, dev,
                                       X, categorical, numeric, frame['ResponseId'],
                                       settings, model_checkpoint, signature)
        variant_rows.append({'family': 'outcome', 'specification': outcome,
                             'predictor_count': len(sc), 'heldout_n': len(test),
                             'selected_iterations': meta['selected_iterations'],
                             **meta['metrics']})
    variant_features = supplementary_features(frame, X, categorical, numeric)
    for variant in ('income', 'pressure', 'age', 'gender'):
        xx, cats, nums = variant_features[variant]
        cols = xx.columns.tolist()
        meta = module.train_and_record('FEATURE_' + variant, cols, primary, fit, val, test, dev,
                                       xx, cats, nums, frame['ResponseId'],
                                       settings, model_checkpoint, signature)
        variant_rows.append({'family': 'feature', 'specification': variant,
                             'predictor_count': len(cols), 'heldout_n': len(test),
                             'selected_iterations': meta['selected_iterations'],
                             **meta['metrics']})

    output = {
        '04_numerical_results/modeling/' + STAGE + '/01_original_eight_sensitivity_model_metrics.csv':
            pd.DataFrame(variant_rows),
        '04_numerical_results/measurement/' + STAGE + '/02_WSQS_weighting_and_domain_exclusion_agreement.csv':
            pd.DataFrame(score_rows),
        '04_numerical_results/measurement/' + STAGE + '/03_Q6_1_satisfaction_vs_WSQS_and_CORE7.csv':
            pd.DataFrame(satisfaction_rows),
        '04_numerical_results/measurement/' + STAGE + '/04_Q6_1_fixed_satisfaction_CORE7A_distribution.csv':
            pd.DataFrame(grouped_rows),
        '04_numerical_results/measurement/' + STAGE + '/05_Q6_1_fixed_satisfaction_domain_profiles.csv':
            pd.DataFrame(profile_rows),
    }
    for relative, table in output.items():
        save_csv(work / relative, table)
    extra_path = measure_dir / '06_population_priority_weights_and_fallback.json'
    atomic_json(extra_path, {'priority_weights_development_only': {str(k): float(v) for k, v in weights.items()},
                             'development_n': 4831, 'fallback_households_for_individual_priority_weights': fallback,
                             'mean_absolute_CORE8_vs_CORE7A': float((primary-core7a).abs().mean()),
                             'spearman_CORE8_vs_CORE7A': c8c7a['spearman_rho'],
                             'independent_satisfaction_validation': False})
    completed = list(output) + [str(extra_path.relative_to(work))]
    record = {'status': 'PASS', 'signature': signature, 'inputs': inputs,
              'n': 6039, 'fitted_models': 8,
              'outcome_sensitivities': 4, 'feature_sensitivities': 4,
              'Q6_1_is_in_CORE8': True,
              'new_independent_criterion_validation_claim': False,
              'output_sha256': {name: digest(work / name) for name in completed},
              'run_dir': str(model_checkpoint)}
    atomic_json(audit_file, record)
    print('=' * 76)
    print('PASS | Stage05 completed: 4 outcome + 4 feature sensitivity models')
    print('PASS | CORE8 / CORE7A / CORE7B satisfaction comparison and fixed-Q6.1 profiles')
    print('CORE8-CORE7A mean absolute score difference:', format((primary-core7a).abs().mean(), '.8f'))
    print('CORE8-CORE7A Spearman:', format(c8c7a['spearman_rho'], '.8f'))
    print('RESULTS:', result_dir)
    print('MEASUREMENT:', measure_dir)
    print('AUDIT:', audit_file)
    print('NO historical results or original data modified')
    print('=' * 76)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, default=Path.home() / 'Desktop/Spring 2026/Chapter 3 MANUAL RERUN')
    ap.add_argument('--stage04-script', type=Path,
                    default=Path.home() / 'Downloads/WSQS_STAGE04_PRIMARY_CROSSDOMAIN_N6039.py')
    args = ap.parse_args()
    run(args.base, args.stage04_script.expanduser().resolve())


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('FAILED |', type(error).__name__, str(error), file=sys.stderr, flush=True)
        raise
