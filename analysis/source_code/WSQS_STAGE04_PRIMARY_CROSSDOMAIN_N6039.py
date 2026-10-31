#!/usr/bin/env python3
"""Stage 04: verified, resumable primary block and cross-domain empirical CatBoost fits.

RUN ON YOUR MAC USING THE ISOLATED ENVIRONMENT:
  WORK="$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_WSQS_RERUN_20261030_N6039"
  "$WORK/.venv_wsqs_n6039/bin/python" ~/Downloads/WSQS_STAGE04_PRIMARY_CROSSDOMAIN_N6039.py

SOURCE OF TRUTH: N6039 locked analytical dataset; October 19 original scientific
configurations and source implementation. No historical input or code overwritten.

23 CatBoost fits: seven nonempty predictor-block coalitions plus eight original
cross-domain and eight matched C+P-only cross-domain models. A mean-only model
is evaluated without fitting, followed by three-block Shapley R2 allocation.

Private predictions and fitted CatBoost models are saved per fit for SHAP,
robustness and exhibit production. Outputs are NOT authorized for public release.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

PROJECT = 'NPJ_WSQS_RERUN_20261030_N6039'
STAGE = 'STAGE04_PRIMARY_CROSSDOMAIN_N6039'
EXPECTED_COUNTS = (6039, 182, 3864, 967, 1208, 4831)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check(cond, message):
    if not cond:
        raise RuntimeError(message)


def json_load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path: Path, data):
    check(not path.exists(), 'Will not overwrite existing output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8', suffix='.partial') as fp:
        temp = Path(fp.name)
        json.dump(data, fp, indent=2, allow_nan=False)
        fp.write('\n')
    os.replace(temp, path)


def csv_once(path: Path, df: pd.DataFrame):
    check(not path.exists(), 'Will not overwrite existing output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8', suffix='.partial') as fp:
        temp = Path(fp.name)
        df.to_csv(fp, index=False, float_format='%.17g')
    os.replace(temp, path)


def train_and_record(key, features, target, fit, val, test, dev,
                     X, cat_all, num_all, ids, settings, checkpoint_root, signature):
    """Mirror original catboost_predictions: validate on fit, then refit on dev."""
    final_dir = checkpoint_root / key
    if final_dir.exists():
        meta = json_load(final_dir / 'model_manifest.json')
        check(meta['signature'] == signature, f'Stale checkpoint for {key}')
        check(meta['features'] == features, f'Changed feature specification for {key}')
        check(meta['fitting_n'] == len(fit) and meta['validation_n'] == len(val)
              and meta['heldout_n'] == len(test), f'Changed partition for {key}')
        for fname in ('final_model.cbm', 'heldout_predictions.csv', 'validation_history.json'):
            item = final_dir / fname
            check(item.is_file() and digest(item) == meta['output_sha256'][fname],
                  f'Corrupt checkpoint {key}/{fname}')
        print(f'REUSE | {key} | R2={meta["metrics"]["r2"]:.9f}', flush=True)
        return meta

    from wsqs.models import prepare_catboost, summary_metrics
    cats = [c for c in features if c in cat_all]
    nums = [c for c in features if c in num_all]
    check(len(cats) + len(nums) == len(features), f'Untyped predictors in {key}')
    check(set(cats).isdisjoint(nums), f'Overlapping feature types in {key}')
    prepared = prepare_catboost(X[features], cats, nums)

    params = dict(settings['catboost'])
    params.update(verbose=False, allow_writing_files=False)
    selection = CatBoostRegressor(**params)
    selection.fit(
        Pool(prepared.iloc[fit], target.iloc[fit], cat_features=cats),
        eval_set=Pool(prepared.iloc[val], target.iloc[val], cat_features=cats),
        use_best_model=True,
    )
    best = selection.get_best_iteration()
    check(best is not None and int(best) >= 0, f'No validation iteration for {key}')
    best_trees = int(best) + 1

    # Exactly reproduce source catboost_predictions: remove early stopping and
    # train the selected number of trees on ALL development observations.
    final_params = dict(params)
    final_params['iterations'] = best_trees
    final_params.pop('od_type', None)
    final_params.pop('od_wait', None)
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(Pool(prepared.iloc[dev], target.iloc[dev], cat_features=cats))
    predicted = np.asarray(final_model.predict(
        Pool(prepared.iloc[test], cat_features=cats)
    ), dtype=float)
    check(predicted.shape == (len(test),) and np.isfinite(predicted).all(),
          f'Invalid predictions from {key}')
    metrics = summary_metrics(target.iloc[test], predicted)
    check(np.isfinite(metrics['rmse_100']) and np.isfinite(metrics['r2']),
          f'Invalid heldout evaluation for {key}')

    tmp = Path(tempfile.mkdtemp(prefix=key + '.partial.', dir=checkpoint_root))
    try:
        final_model.save_model(str(tmp / 'final_model.cbm'))
        detail = pd.DataFrame({
            'canonical_row_index': test.astype(int),
            'ResponseId': ids.iloc[test].astype(str).to_numpy(),
            'observed_WSQS_100': target.iloc[test].to_numpy(dtype=float),
            'predicted_WSQS_100': predicted,
            'residual_observed_minus_predicted': target.iloc[test].to_numpy(dtype=float) - predicted,
        })
        detail.to_csv(tmp / 'heldout_predictions.csv', index=False, float_format='%.17g')
        (tmp / 'validation_history.json').write_text(
            json.dumps(selection.get_evals_result(), indent=2, allow_nan=False) + '\n', encoding='utf-8')
        output_hashes = {
            filename: digest(tmp / filename)
            for filename in ('final_model.cbm', 'heldout_predictions.csv', 'validation_history.json')
        }
        meta = {
            'model_id': key, 'signature': signature, 'features': features,
            'categorical': cats, 'numeric': nums,
            'fitting_n': len(fit), 'validation_n': len(val),
            'development_n': len(dev), 'heldout_n': len(test),
            'selected_iterations': best_trees, 'metrics': metrics,
            'output_sha256': output_hashes,
            'fitting_procedure': 'validation on fitting; best iteration; refit on full development; evaluate untouched held-out',
            'original_catboost_configuration': settings['catboost'],
            'outcome_scale': '0-100; higher means better service',
            'private_respondent_outputs': True,
        }
        (tmp / 'model_manifest.json').write_text(
            json.dumps(meta, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(tmp, final_dir)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(f'PASS | {key} | R2={metrics["r2"]:.9f} | RMSE={metrics["rmse_100"]:.9f} | trees={best_trees}', flush=True)
    return meta


def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, default=Path.home() / 'Desktop/Spring 2026/Chapter 3 MANUAL RERUN')
    args = ap.parse_args()
    work = args.base.expanduser().resolve() / PROJECT
    data_path = work / '00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    lock_path = work / '07_audit/COHORT_DATASET_LOCK_N6039.json'
    preflight_path = work / '04_numerical_results/measurement/CORE8_AND_FEATURE_PREFLIGHT_N6039.json'
    env_path = work / '07_audit/SCIENTIFIC_ENVIRONMENT_N6039.json'
    code = work / '02_code_private/reconciled/WSQS_Study_Repro_N6039'
    expected_python = work / '.venv_wsqs_n6039/bin/python'
    check(Path(sys.executable).resolve() == expected_python.resolve(),
          'Use isolated WSQS Python, not Anaconda base: ' + str(expected_python))
    for item in (data_path, lock_path, preflight_path, env_path, code / 'config/models.json',
                 code / 'config/scientific_plans.json'):
        check(item.is_file(), 'Missing prerequisite ' + str(item))
    lock, preflight, env = map(json_load, (lock_path, preflight_path, env_path))
    data_sha = digest(data_path)
    check(data_sha == lock['output_sha256']['primary'] == preflight['source_sha256'],
          'Frozen source dataset SHA256 does not match cohort/preflight lock')
    check(env.get('status', '').lower() != 'failed', 'Environment audit failed')

    sys.path.insert(0, str(code / 'src'))
    from wsqs.extended import BLOCK_KEYS, DOMAINS, block_columns, block_shapley, scientific_plans
    from wsqs.features import construct_features
    from wsqs.models import model_settings, partition_indices, summary_metrics

    plans = scientific_plans()
    settings = model_settings()
    check(int(settings['primary_seed']) == 20260714, 'Primary seed has changed')
    check(int(settings['catboost']['random_seed']) == 20260714, 'CatBoost random seed has changed')

    script_sha = digest(Path(__file__).resolve())
    scientific_code_files = [code / 'src/wsqs/models.py', code / 'src/wsqs/features.py',
                             code / 'src/wsqs/extended.py']
    sources = {str(p.relative_to(work)): digest(p) for p in
               [code / 'config/models.json', code / 'config/scientific_plans.json',
                code / 'config/feature_spec.json', *scientific_code_files]}
    fingerprint = {
        'source_cohort_sha256': data_sha,
        'scientific_sources_sha256': sources,
        'stage04_script_sha256': script_sha,
        'python_version': sys.version.split()[0],
        'seed': int(settings['primary_seed']),
    }
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()

    print('=' * 75)
    print('STAGE 04 | ORIGINAL 7 BLOCK MODELS + 8 ORIGINAL AND 8 NEW CROSS-DOMAIN FITS')
    print('SOURCE SHA256:', data_sha)
    print('RUN SIGNATURE:', signature)
    print('=' * 75, flush=True)

    frame = pd.read_csv(data_path, low_memory=False, dtype={'ResponseId': str})
    check(frame.shape == EXPECTED_COUNTS[:2], 'Source dimensions changed')
    check(frame.ResponseId.is_unique and frame.ResponseId.notna().all(), 'ResponseId integrity failure')
    if 'municipal_supply_applicable' not in frame.columns:
        check('source09M_municipal_eligible' in frame.columns, 'Missing provider-applicability field')
        frame = frame.rename(columns={'source09M_municipal_eligible': 'municipal_supply_applicable'})
    X, cats, nums = construct_features(frame)
    check(X.shape == (6039, 33) and list(X.columns) == plans['full'],
          'Wrong primary predictor specification')
    check(len(plans['blocks']['S']) == 22 and len(plans['blocks']['C']) == 3
          and len(plans['blocks']['P']) == 8, 'Wrong predictor block membership')

    raw = frame[[f'{d}_raw_severity' for d in DOMAINS]].apply(pd.to_numeric, errors='coerce')
    check(raw.notna().sum(axis=1).between(7, 8).all(), 'CORE domain-completeness changed')
    check(raw.stack().isin([1, 2, 3, 4, 5]).all(), 'CORE domain severity coding changed')
    target = pd.Series((5 - raw.mean(axis=1)) * 25, dtype=float)
    check(target.notna().all() and target.between(0, 100).all(), 'CORE8 target invalid')
    check(np.isclose(float(target.mean()), float(preflight['core8_mean_full_precision']), rtol=0, atol=1e-11),
          'CORE8 mean differs from Stage 03')

    fit, val, test, dev = partition_indices(len(frame), int(settings['primary_seed']))
    check((len(frame), frame.shape[1], len(fit), len(val), len(test), len(dev)) == EXPECTED_COUNTS,
          'Partition dimensions changed')

    checkpoint_root = work / '03_run/checkpoints' / STAGE
    results_root = work / '04_numerical_results/modeling' / STAGE
    audit_path = work / '07_audit' / (STAGE + '_COMPLETE.json')
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    complete = audit_path.exists()
    if complete:
        audit = json_load(audit_path)
        check(audit['signature'] == signature, 'Stage 04 result signature differs; refusing stale result')
        print('PASS | Stage 04 already complete for exact same code/data; nothing to refit')
        print('RESULTS:', results_root)
        return

    ids = frame['ResponseId']
    split_path = checkpoint_root / 'frozen_primary_partition_private.csv'
    if not split_path.exists():
        parts = np.full(len(frame), '', dtype=object)
        parts[fit], parts[val], parts[test] = 'fitting', 'validation', 'heldout_test'
        check(np.all(parts != ''), 'Missing partition memberships')
        csv_once(split_path, pd.DataFrame({
            'canonical_row_index': np.arange(len(frame)),
            'ResponseId': ids,
            'partition': parts,
        }))
    else:
        saved = pd.read_csv(split_path, dtype={'ResponseId': str})
        check(saved['ResponseId'].tolist() == ids.astype(str).tolist(),
              'Saved partition IDs differ from frozen dataset')
        expected_parts = np.full(len(frame), '', dtype=object)
        expected_parts[fit], expected_parts[val], expected_parts[test] = 'fitting', 'validation', 'heldout_test'
        check(np.array_equal(saved['partition'].to_numpy(), expected_parts),
              'Saved partition differs from original seed')

    r2 = {}
    rows = []
    mean_pred = np.full(len(test), float(target.iloc[dev].mean()))
    mean_metrics = summary_metrics(target.iloc[test], mean_pred)
    r2[''] = mean_metrics['r2']
    rows.append({'block': 'Mean baseline', 'n_predictors': 0, 'selected_iterations': None,
                 'fit_n': len(fit), 'validation_n': len(val), 'development_n': len(dev),
                 'heldout_n': len(test), **mean_metrics})

    for block in BLOCK_KEYS:
        cols = block_columns(block, plans)
        check(len(cols) == sum(len(plans['blocks'][b]) for b in block), f'Block {block} mismatch')
        meta = train_and_record('BLOCK_' + block, cols, target, fit, val, test, dev,
                                X, cats, nums, ids, settings, checkpoint_root, signature)
        r2[block] = meta['metrics']['r2']
        rows.append({'block': block, 'n_predictors': len(cols), 'selected_iterations': meta['selected_iterations'],
                     'fit_n': len(fit), 'validation_n': len(val), 'development_n': len(dev),
                     'heldout_n': len(test), **meta['metrics']})

    allocation = block_shapley(r2)
    gain = r2['SCP'] - r2['']
    check(np.isclose(sum(allocation.values()), gain, atol=1e-10, rtol=0), 'Block Shapley sum mismatch')
    shapley = pd.DataFrame([{'block': b, 'allocated_r2': allocation[b],
                             'share_percent': 100 * allocation[b] / gain if gain else np.nan,
                             'full_minus_mean_r2': gain} for b in ('S', 'C', 'P')])
    print('PASS | Original 7 coalitions, mean baseline and block Shapley reconciled', flush=True)

    pairs = []
    cp = [c for c in plans['full'] if c in set(plans['blocks']['C'] + plans['blocks']['P'])]
    check(len(cp) == 11, 'C+P baseline must contain 11 predictors')
    for domain in DOMAINS:
        raw_domain = pd.to_numeric(frame[f'{domain}_raw_severity'], errors='coerce')
        ydomain = pd.Series((5.0 - raw_domain) * 25.0, dtype=float)
        dfit = fit[ydomain.iloc[fit].notna().to_numpy()]
        dval = val[ydomain.iloc[val].notna().to_numpy()]
        dtest = test[ydomain.iloc[test].notna().to_numpy()]
        ddev = np.sort(np.r_[dfit, dval])
        check(min(map(len, (dfit, dval, dtest))) > 0, 'Empty cross-domain partition ' + domain)
        full_cols = plans['cross_domain'][domain]
        check(set(cp).issubset(full_cols), f'Cross-domain {domain} excludes C or P')
        check(len(full_cols) > len(cp), f'Cross-domain {domain} unexpectedly contains no service variables')
        original = train_and_record('XDOMAIN_FULL_' + domain, full_cols, ydomain,
                                    dfit, dval, dtest, ddev, X, cats, nums, ids,
                                    settings, checkpoint_root, signature)
        restricted = train_and_record('XDOMAIN_CP_' + domain, cp, ydomain,
                                      dfit, dval, dtest, ddev, X, cats, nums, ids,
                                      settings, checkpoint_root, signature)
        check(original['heldout_n'] == restricted['heldout_n'], 'Unmatched heldout sample')
        row = {'domain': domain, 'heldout_n': len(dtest),
               'development_n': len(ddev), 'fitting_n': len(dfit), 'validation_n': len(dval),
               'full_feature_n': len(full_cols), 'cp_feature_n': len(cp),
               'full_r2': original['metrics']['r2'], 'cp_r2': restricted['metrics']['r2'],
               'incremental_other_domain_service_support_r2': original['metrics']['r2'] - restricted['metrics']['r2'],
               'full_rmse_100': original['metrics']['rmse_100'], 'cp_rmse_100': restricted['metrics']['rmse_100'],
               'full_mae_100': original['metrics']['mae_100'], 'cp_mae_100': restricted['metrics']['mae_100'],
               'full_spearman': original['metrics']['spearman'], 'cp_spearman': restricted['metrics']['spearman'],
               'full_selected_iterations': original['selected_iterations'], 'cp_selected_iterations': restricted['selected_iterations']}
        pairs.append(row)
        print(f'PASS | Paired {domain}: full R2={row["full_r2"]:.8f}; CP R2={row["cp_r2"]:.8f}; delta={row["incremental_other_domain_service_support_r2"]:.8f}', flush=True)

    csv_once(results_root / '01_primary_block_heldout_metrics.csv', pd.DataFrame(rows))
    csv_once(results_root / '02_three_block_shapley_r2.csv', shapley)
    csv_once(results_root / '03_cross_domain_full_vs_CP_heldout.csv', pd.DataFrame(pairs))
    # These aggregate results may support publication; underlying checkpoint models
    # and ResponseId-tagged predictions remain private.
    audit = {'status': 'PASS', 'signature': signature, 'fingerprint': fingerprint,
             'households': len(frame), 'primary_coalition_models': 7,
             'crossdomain_original_models': 8, 'crossdomain_new_CP_baselines': 8,
             'mean_baseline': 1, 'total_empirical_catboost_fits': 23,
             'heldout_split_sizes': {'fit': len(fit), 'validation': len(val), 'test': len(test), 'development': len(dev)},
             'output_sha256': {p.name: digest(p) for p in results_root.glob('*.csv')},
             'checkpoint_dir': str(checkpoint_root), 'results_dir': str(results_root),
             'no_causal_interpretation': True, 'private_predictions_present': True}
    atomic_json(audit_path, audit)
    print('=' * 75)
    print('PASS | STAGE 04 COMPLETED: 7 original blocks + 16 matched cross-domain models')
    print('BLOCK SHAPLEY AND CROSS-DOMAIN R2 TABLES:', results_root)
    print('FITTED CBM MODELS AND HELDOUT PREDICTIONS:', checkpoint_root)
    print('AUDIT:', audit_path)
    print('NEXT | Stage 05: outcome and feature sensitivity, ten-split models, SHAP, priority null, B200')
    print('=' * 75)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('FAILED |', type(exc).__name__, str(exc), file=sys.stderr, flush=True)
        raise
