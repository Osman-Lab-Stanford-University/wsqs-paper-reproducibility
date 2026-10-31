#!/usr/bin/env python3
"""Stage 08 | Frozen N6039 CatBoost SHAP, all 33 features and B1000 uncertainty.

Execute on the author's Mac after Stages 01–07:

  WORK="$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_WSQS_RERUN_20261030_N6039"
  "$WORK/.venv_wsqs_n6039/bin/python" \
    ~/Downloads/WSQS_STAGE08_SHAP_ATTRIBUTION_N6039.py

Scientific protocol:
  - NEVER fit, tune or mutate a model in this stage.
  - Use frozen Stage 04 BLOCK_SCP final CatBoost (trained on all development
    observations with the validation-selected number of trees).
  - Calculate signed CatBoost ShapValues for all 1,208 held-out households.
  - Check signed SHAP additivity against saved frozen model predictions.
  - Group mean absolute SHAP by the preregistered 22 S / 3 C / 8 P blocks.
  - B=1000 independent-with-replacement held-out household resamples,
    conditional on the unchanged frozen model and explanations. Not refits.
  - Save full-precision aggregate results for main Tables 5–6/Figure 6 and
    Supplementary Table S26/Figure S6. Save private per-household feature and
    signed-SHAP data for later plotting (not for public release).
  - Do not confuse conditional SHAP intervals with independent-study or
    model-refit confidence intervals, or grouped SHAP with block Shapley R².

Original data, historic scripts and previous stage outputs are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

PROJECT = 'NPJ_WSQS_RERUN_20261030_N6039'
STAGE = 'STAGE08_SHAP_ATTRIBUTION_N6039'
EXPECTED_COHORT_SHA = 'c72b8480d3cff223d1d4cf0ebcc29a3dd80204d563cd63a4295f6739f0ed77a1'
EXPECTED_N = 6039
EXPECTED_COLS = 182
EXPECTED_TEST_N = 1208
B = 1000
BOOTSTRAP_SEED = 20260714
FEATURE_BLOCKS = ('S', 'C', 'P')


def check(test, reason):
    if not test:
        raise RuntimeError(reason)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o).__name__)


def write_new_json(path: Path, payload):
    check(not path.exists(), 'Output exists, refusing overwrite: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.partial', encoding='utf-8', delete=False) as f:
        tmppath = Path(f.name)
        json.dump(payload, f, indent=2, allow_nan=False, default=json_default)
        f.write('\n')
    os.replace(tmppath, path)


def write_new_csv(path: Path, data):
    check(not path.exists(), 'Output exists, refusing overwrite: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.partial', encoding='utf-8', delete=False) as f:
        tmppath = Path(f.name)
        data.to_csv(f, index=False, float_format='%.17g')
    os.replace(tmppath, path)


def group_features(features, plans):
    """Return stable column indices for mutually exclusive pre-existing blocks."""
    block_indices = {}
    for key in FEATURE_BLOCKS:
        members = plans['blocks'][key]
        block_indices[key] = np.asarray([i for i, f in enumerate(features) if f in set(members)], dtype=int)
        check(set(features[i] for i in block_indices[key]) == set(members),
              'Missing or extra members in block ' + key)
    lengths = [len(block_indices[b]) for b in FEATURE_BLOCKS]
    check(lengths == [22, 3, 8], 'Unexpected group sizes: ' + str(lengths))
    check(sorted(np.concatenate(list(block_indices.values())).tolist()) == list(range(len(features))),
          'Predictors not partitioned exactly once')
    return block_indices


def calculate_shap_from_model(model, prepared_test, cats, predictions):
    """CatBoost ShapValues returns n x (p+1) matrix, final column = baseline."""
    pool = Pool(prepared_test, cat_features=cats)
    full = np.asarray(model.get_feature_importance(data=pool, type='ShapValues'), dtype=float)
    check(full.shape == (len(prepared_test), prepared_test.shape[1] + 1),
          'Invalid CatBoost SHAP matrix shape: ' + str(full.shape))
    check(np.isfinite(full).all(), 'Nonfinite SHAP values')
    contributions, baseline = full[:, :-1], full[:, -1]
    predicted_from_shap = contributions.sum(axis=1) + baseline
    max_error = float(np.max(np.abs(predicted_from_shap - predictions)))
    check(max_error < 1e-5, f'SHAP additivity did not reconcile (max error={max_error})')
    return contributions, baseline, max_error


def shaps_for_bootstrap(shap_values, block_idx, b=B, seed=BOOTSTRAP_SEED):
    """Draw held-out households with replacement; no model refit.

    For each sample calculate feature mean absolute contributions, block-level
    absolute contributions, and block shares of the total mean absolute SHAP.
    """
    n, p = shap_values.shape
    check(n > 2 and p == 33, 'Expected 33 predictor SHAP matrix')
    absolute = np.abs(shap_values)
    rng = np.random.default_rng(seed)
    feature_boot = np.empty((b, p), dtype=float)
    group_abs_boot = np.empty((b, len(FEATURE_BLOCKS)), dtype=float)
    share_boot = np.empty_like(group_abs_boot)
    for i in range(b):
        draw = rng.integers(0, n, size=n, endpoint=False)
        feature_boot[i, :] = absolute[draw, :].mean(axis=0)
        for j, group in enumerate(FEATURE_BLOCKS):
            group_abs_boot[i, j] = feature_boot[i, block_idx[group]].sum()
        denominator = group_abs_boot[i, :].sum()
        check(denominator > 0 and np.isfinite(denominator), 'Invalid bootstrap total SHAP')
        share_boot[i, :] = 100 * group_abs_boot[i, :] / denominator
    check(np.allclose(share_boot.sum(axis=1), 100, rtol=0, atol=1e-10),
          'Bootstrap group shares fail to sum to 100')
    return feature_boot, group_abs_boot, share_boot


def result_frames(contributions, features, block_idx, feat_boot, abs_boot, share_boot):
    abs_values = np.abs(contributions)
    avg_abs = abs_values.mean(axis=0)
    avg_signed = contributions.mean(axis=0)
    block_abs = {name: float(avg_abs[idx].sum()) for name, idx in block_idx.items()}
    total = sum(block_abs.values())
    check(total > 0, 'No SHAP contribution to aggregate')
    group_rows = []
    for j, name in enumerate(FEATURE_BLOCKS):
        abs_ci = np.quantile(abs_boot[:, j], [0.025, 0.975])
        share_ci = np.quantile(share_boot[:, j], [0.025, 0.975])
        group_rows.append({
            'block': name,
            'number_of_predictors': len(block_idx[name]),
            'heldout_n': len(contributions),
            'mean_absolute_shap_wsqs_points': block_abs[name],
            'bootstrap_mean_abs_ci2p5': float(abs_ci[0]),
            'bootstrap_mean_abs_ci97p5': float(abs_ci[1]),
            'percent_of_total_absolute_shap': 100 * block_abs[name] / total,
            'bootstrap_share_ci2p5_percent': float(share_ci[0]),
            'bootstrap_share_ci97p5_percent': float(share_ci[1]),
            'resamples': len(share_boot),
            'uncertainty_scope': 'heldout-household resampling; frozen model',
        })
    group_df = pd.DataFrame(group_rows)
    check(np.isclose(group_df['percent_of_total_absolute_shap'].sum(), 100, atol=1e-10),
          'Group shares not equal to 100')
    predictor_group = {features[i]: g for g, inds in block_idx.items() for i in inds}
    rows = []
    for i, feature in enumerate(features):
        low, high = np.quantile(feat_boot[:, i], [0.025, 0.975])
        rows.append({
            'feature': feature,
            'block': predictor_group[feature],
            'heldout_n': len(contributions),
            'mean_abs_shap_wsqs_points': float(avg_abs[i]),
            'mean_signed_shap_wsqs_points': float(avg_signed[i]),
            'percent_of_total_abs_shap': 100 * float(avg_abs[i]) / total,
            'bootstrap_mean_abs_ci2p5': float(low),
            'bootstrap_mean_abs_ci97p5': float(high),
        })
    feature_df = pd.DataFrame(rows).sort_values(
        ['mean_abs_shap_wsqs_points', 'feature'], ascending=[False, True]
    ).reset_index(drop=True)
    feature_df.insert(0, 'rank', np.arange(1, len(feature_df) + 1))
    feature_df['cumulative_percent_of_total_abs_shap'] = feature_df['percent_of_total_abs_shap'].cumsum()
    check(abs(float(feature_df['percent_of_total_abs_shap'].sum()) - 100) < 1e-8,
          'Per-feature total SHAP shares not equal to 100')
    return group_df, feature_df


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path.home() / 'Desktop/Spring 2026/Chapter 3 MANUAL RERUN')
    args = parser.parse_args()
    work = args.base.expanduser().resolve() / PROJECT
    actual_python = Path(sys.executable).resolve()
    venv_python = (work / '.venv_wsqs_n6039/bin/python').resolve()
    check(actual_python == venv_python, 'Use the isolated scientific WSQS interpreter: ' + str(venv_python))

    data_path = work / '00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    cohort_audit = work / '07_audit/COHORT_DATASET_LOCK_N6039.json'
    stage04_audit = work / '07_audit/STAGE04_PRIMARY_CROSSDOMAIN_N6039_COMPLETE.json'
    model_dir = work / '03_run/checkpoints/STAGE04_PRIMARY_CROSSDOMAIN_N6039/BLOCK_SCP'
    model_file = model_dir / 'final_model.cbm'
    model_meta_file = model_dir / 'model_manifest.json'
    pred_file = model_dir / 'heldout_predictions.csv'
    code = work / '02_code_private/reconciled/WSQS_Study_Repro_N6039'
    plans_file = code / 'config/scientific_plans.json'
    check(all(p.is_file() for p in (data_path, cohort_audit, stage04_audit, model_file, model_meta_file,
                                    pred_file, plans_file, code/'src/wsqs/models.py',code/'src/wsqs/features.py')),
          'Missing prerequisite from frozen cohort or Stage 04')

    data_sha = sha256(data_path)
    lock = read_json(cohort_audit)
    check(data_sha == EXPECTED_COHORT_SHA == lock['output_sha256']['primary'], 'N6039 source SHA changed')
    original_audit = read_json(stage04_audit)
    model_meta = read_json(model_meta_file)
    check(original_audit['status'] == 'PASS', 'Stage 04 audit not passed')
    check(model_meta['signature'] == original_audit['signature'], 'Stage 04 model/run signatures differ')
    check(model_meta['heldout_n'] == EXPECTED_TEST_N, 'Wrong frozen test sample')
    for fn, p in [('final_model.cbm', model_file), ('heldout_predictions.csv', pred_file),
                  ('validation_history.json', model_dir / 'validation_history.json')]:
        check(sha256(p) == model_meta['output_sha256'][fn], 'Stage 04 checkpoint corrupted: ' + fn)

    source_hashes = {str(p.relative_to(work)): sha256(p) for p in
                     [plans_file, code / 'config/models.json', code / 'config/feature_spec.json',
                      code / 'src/wsqs/models.py', code / 'src/wsqs/features.py',
                      code / 'src/wsqs/extended.py']}
    signature_components = {
        'data_sha256': data_sha,
        'stage04_run_signature': original_audit['signature'],
        'frozen_model_sha256': sha256(model_file),
        'original_model_manifest_sha256': sha256(model_meta_file),
        'scientific_code_sha256': source_hashes,
        'stage08_script_sha256': sha256(Path(__file__).resolve()),
        'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap_resamples': B,
    }
    signature = hashlib.sha256(json.dumps(signature_components, sort_keys=True).encode('utf-8')).hexdigest()

    checkpoint_dir = work / '03_run/checkpoints' / STAGE
    results_dir = work / '04_numerical_results/inference' / STAGE
    audit_path = work / '07_audit' / (STAGE + '_COMPLETE.json')
    output_names = [
        '01_grouped_shap_with_bootstrap_ci.csv',
        '02_all_33_predictors_shap_with_bootstrap_ci.csv',
        '03_top15_shap_predictors.csv',
        '04_bootstrap_group_distribution_B1000.csv',
        '05_shap_additivity_and_sources.json',
    ]
    if audit_path.exists():
        old = read_json(audit_path)
        check(old['signature'] == signature and old['status'] == 'PASS', 'Stale/incomplete Stage 08 audit')
        for name, expected in old['output_sha256'].items():
            p = results_dir / name
            check(p.is_file() and sha256(p) == expected, 'Stage 08 aggregate output changed: ' + name)
        for name, expected in old['checkpoint_sha256'].items():
            p = checkpoint_dir / name
            check(p.is_file() and sha256(p) == expected, 'Stage 08 checkpoint changed: ' + name)
        print('PASS | Stage 08 complete and verified; no recomputation')
        print('RESULTS:', results_dir)
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    check(not any((results_dir / f).exists() for f in output_names),
          'Partial Stage 08 aggregate files exist without completed audit; inspect before retrying')

    sys.path.insert(0, str(code / 'src'))
    from wsqs.features import construct_features
    from wsqs.models import model_settings, partition_indices, prepare_catboost
    from wsqs.extended import scientific_plans
    settings = model_settings()
    plans = scientific_plans()
    check(int(settings['primary_seed']) == 20260714, 'Changed original split seed')

    print('=' * 77)
    print('STAGE 08 | FROZEN FULL CATBOOST SHAP + 1000 CONDITIONAL BOOTSTRAPS', flush=True)
    print('SOURCE SHA256:', data_sha)
    print('RUN SIGNATURE:', signature)
    print('FROZEN MODEL SHA256:', sha256(model_file))
    print('=' * 77, flush=True)

    df = pd.read_csv(data_path, low_memory=False, dtype={'ResponseId': str})
    check(df.shape == (EXPECTED_N, EXPECTED_COLS), 'Changed dataset shape')
    check(df.ResponseId.notna().all() and df.ResponseId.is_unique, 'Invalid ResponseIds')
    if 'municipal_supply_applicable' not in df.columns:
        check('source09M_municipal_eligible' in df.columns, 'Missing municipal use applicability')
        df = df.rename(columns={'source09M_municipal_eligible': 'municipal_supply_applicable'})
    X, categorical, numeric = construct_features(df)
    features = plans['full']
    check(list(X.columns) == features == model_meta['features'], '33 predictors differ from frozen model')
    check(len(features) == 33 and X.shape == (EXPECTED_N, 33), 'Unexpected model features')
    check(set(categorical) == set(model_meta['categorical']) and
          set(numeric) == set(model_meta['numeric']), 'Categorical/numeric columns changed')
    blocks = group_features(features, plans)
    fit, validation, test, dev = partition_indices(len(df), 20260714)
    check((len(fit), len(validation), len(test), len(dev)) == (3864,967,1208,4831),
          'Partition changed')
    predictions_df = pd.read_csv(pred_file, dtype={'ResponseId': str})
    check(len(predictions_df) == len(test), 'Stage 04 held-out predictions changed')
    check(np.array_equal(predictions_df['canonical_row_index'].to_numpy(dtype=int), test),
          'Stage 04 heldout partition/order changed')
    check(predictions_df.ResponseId.astype(str).tolist() == df.ResponseId.iloc[test].astype(str).tolist(),
          'Stage 04 heldout ResponseIds differ')

    prepared = prepare_catboost(X[features], model_meta['categorical'], model_meta['numeric'])
    model = CatBoostRegressor()
    model.load_model(str(model_file))
    check(model.tree_count_ == model_meta['selected_iterations'], 'Frozen tree count differs')
    on_test = prepared.iloc[test]
    new_predictions = np.asarray(model.predict(Pool(on_test, cat_features=model_meta['categorical'])), dtype=float)
    recorded_predictions = predictions_df['predicted_WSQS_100'].to_numpy(dtype=float)
    prediction_diff = float(np.max(np.abs(new_predictions - recorded_predictions)))
    check(prediction_diff < 1e-7, 'Frozen model failed heldout prediction identity check')

    print(f'PASS | Reused Stage04 BLOCK_SCP model; {len(test)} unchanged heldout households', flush=True)
    print('CALCULATE | CatBoost native SHAP for 33 predictors (no model fitting)', flush=True)
    shap, expected, additivity_error = calculate_shap_from_model(
        model, on_test, model_meta['categorical'], new_predictions)
    print(f'PASS | SHAP exact additive reconciliation; max error={additivity_error:.3g}', flush=True)

    private_output_name = 'heldout_shap_and_predictors_PRIVATE.csv'
    private_path = checkpoint_dir / private_output_name
    checkpoint_note = checkpoint_dir / 'SHAP_EXTRACTION_LOCK.json'
    if private_path.exists():
        check(checkpoint_note.is_file(), 'Private SHAP output exists without lock')
        stored_note = read_json(checkpoint_note)
        check(stored_note['signature'] == signature and
              stored_note['private_sha256'] == sha256(private_path), 'Stale/corrupt private SHAP checkpoint')
        print('PASS | Existing private SHAP output verified')
    else:
        private = pd.DataFrame({
            'canonical_row_index': test,
            'ResponseId': df.ResponseId.iloc[test].astype(str).to_numpy(),
            'observed_WSQS_100': predictions_df['observed_WSQS_100'].to_numpy(dtype=float),
            'predicted_WSQS_100': new_predictions,
            'catboost_shap_expected_value_wsqs_points': expected,
        })
        for k, feature in enumerate(features):
            private['shap_' + feature] = shap[:, k]
        for feature in features:
            private['value_' + feature] = X[feature].iloc[test].reset_index(drop=True)
        write_new_csv(private_path, private)
        write_new_json(checkpoint_note, {
            'signature': signature, 'private_file': private_output_name,
            'private_sha256': sha256(private_path), 'rows': len(private),
            'contains_ResponseId_and_household_values': True,
        })
    print('PASS | Private, full-precision row-level SHAP and feature data saved', flush=True)

    print(f'BOOTSTRAP | B={B} heldout-household resamples; frozen model', flush=True)
    fboot, aboot, sboot = shaps_for_bootstrap(shap, blocks)
    group_df, feature_df = result_frames(shap, features, blocks, fboot, aboot, sboot)
    top15 = feature_df.head(15).copy()

    bootstrap = pd.DataFrame({'replicate': np.arange(1,B+1, dtype=int)})
    for j, key in enumerate(FEATURE_BLOCKS):
        bootstrap[f'{key}_mean_abs_shap_wsqs_points'] = aboot[:, j]
        bootstrap[f'{key}_percent_abs_shap'] = sboot[:, j]

    # Write last in a single uninterrupted, separately audited phase.
    write_new_csv(results_dir / output_names[0], group_df)
    write_new_csv(results_dir / output_names[1], feature_df)
    write_new_csv(results_dir / output_names[2], top15)
    write_new_csv(results_dir / output_names[3], bootstrap)
    write_new_json(results_dir / output_names[4], {
        'signature': signature, 'model_id': 'BLOCK_SCP',
        'source_data_sha256': data_sha, 'model_sha256': sha256(model_file),
        'heldout_n': len(test), 'feature_n': len(features),
        'shap_additivity_max_abs_error': additivity_error,
        'frozen_prediction_max_abs_error': prediction_diff,
        'original_model_tree_count': model.tree_count_,
        'total_mean_abs_shap_wsqs_points': float(np.abs(shap).mean(axis=0).sum()),
        'top15_percent_total_abs_shap': float(top15['percent_of_total_abs_shap'].sum()),
        'bootstrap': {'resamples':B, 'seed':BOOTSTRAP_SEED,
                      'sampling':'heldout households with replacement',
                      'model_refit': False},
        'note': 'Grouped absolute SHAP, not three-block Shapley R2; attribution not causal',
    })

    output_hashes = {name: sha256(results_dir / name) for name in output_names}
    checkpoints_hashes = {name: sha256(checkpoint_dir / name) for name in [private_output_name, 'SHAP_EXTRACTION_LOCK.json']}
    audit = {
        'status': 'PASS', 'signature': signature,
        'signature_components': signature_components,
        'source_households': EXPECTED_N, 'heldout_households': EXPECTED_TEST_N,
        'feature_count': len(features), 'groups': {k:len(v) for k,v in blocks.items()},
        'frozen_model_reused': True, 'new_model_fits': 0,
        'bootstrap_resamples': B, 'bootstrap_seed': BOOTSTRAP_SEED,
        'source_stage04_model': str(model_file),
        'output_sha256': output_hashes, 'checkpoint_sha256': checkpoints_hashes,
        'data_privacy': 'Respondent-level features, IDs, and SHAP held only in private project checkpoint; aggregate results only in inference folder',
    }
    write_new_json(audit_path, audit)

    print('=' * 77)
    print('PASS | Stage08 complete: 33 SHAP features, 3 groups, B1000 conditional bootstraps')
    for _, row in group_df.iterrows():
        print(f'{row["block"]}: mean_abs={row["mean_absolute_shap_wsqs_points"]:.6f} | '
              f'share={row["percent_of_total_absolute_shap"]:.3f}% | '
              f'CI=[{row["bootstrap_share_ci2p5_percent"]:.3f}, {row["bootstrap_share_ci97p5_percent"]:.3f}]%')
    print('Top 15 share:', f'{top15["percent_of_total_abs_shap"].sum():.3f}%')
    print('RESULTS:', results_dir)
    print('PRIVATE SHAP SOURCE:', private_path)
    print('AUDIT:', audit_path)
    print('NEXT | Stage09 service-response profiles and 200 development refits')
    print('=' * 77)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('FAILED |', type(exc).__name__, str(exc), file=sys.stderr, flush=True)
        raise
