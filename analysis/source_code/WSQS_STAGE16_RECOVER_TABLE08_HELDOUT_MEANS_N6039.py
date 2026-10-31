#!/usr/bin/env python3
"""Stage 16: recover the six missing Table 8 held-out means, read-only source analysis.

Uses exactly three pre-existing prediction checkpoints:
  Stage04/BLOCK_SC (equal-weight CORE-8, 25 predictors),
  Stage05/OUTCOME_POPULATION_PRIORITY (25 predictors),
  Stage05/OUTCOME_INDIVIDUAL_PRIORITY (25 predictors).

No CatBoost imports or model fitting. Verifies frozen cohort, source-stage audits,
checkpoint manifests/hashes, same 1,208 held-out household identities and order,
and independently recalculates and verifies the model performance metrics.
Only aggregate values (no rows or IDs) are saved.

Mac invocation:
  WORK="$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_WSQS_RERUN_20261030_N6039"
  "$WORK/.venv_wsqs_n6039/bin/python" \\
    ~/Downloads/WSQS_STAGE16_RECOVER_TABLE08_HELDOUT_MEANS_N6039.py
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
from scipy.stats import spearmanr

PROJECT = 'NPJ_WSQS_RERUN_20261030_N6039'
SOURCE_SHA = 'c72b8480d3cff223d1d4cf0ebcc29a3dd80204d563cd63a4295f6739f0ed77a1'
STAGE04 = 'STAGE04_PRIMARY_CROSSDOMAIN_N6039'
STAGE05 = 'STAGE05_SENSITIVITY_SATISFACTION_N6039'
OUTSTAGE = 'STAGE16_TABLE08_HELDOUT_MEANS_N6039'
EXPECTED_N = 1208
PREDICTION_FIELDS = ['canonical_row_index', 'ResponseId', 'observed_WSQS_100',
                     'predicted_WSQS_100', 'residual_observed_minus_predicted']
CONFIG = [
    ('Equal-weight CORE-8', STAGE04, 'BLOCK_SC', 'SC'),
    ('Population-priority CORE-8', STAGE05, 'OUTCOME_POPULATION_PRIORITY', 'POPULATION_PRIORITY'),
    ('Individual-priority CORE-8', STAGE05, 'OUTCOME_INDIVIDUAL_PRIORITY', 'INDIVIDUAL_PRIORITY'),
]


def check(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def jread(path: Path) -> dict:
    check(path.is_file(), f'Missing required file: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def require_hash(path: Path, expected: str) -> None:
    check(path.is_file() and sha(path) == expected, f'File missing/changed: {path}')


def metrics_from_arrays(y, pred) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    check(y.shape == pred.shape and y.ndim == 1 and len(y) >= 3, 'Invalid score/prediction vectors')
    check(np.isfinite(y).all() and np.isfinite(pred).all(), 'Nonfinite observed/predicted values')
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    check(ss_tot > 0, 'Observed outcome variance is zero')
    return {
        'rmse_100': float(np.sqrt(np.mean((y - pred)**2))),
        'mae_100': float(np.mean(np.abs(y - pred))),
        'r2': float(1.0 - ss_res / ss_tot),
        'spearman': float(spearmanr(y, pred).statistic),
    }


def inspect_checkpoint(work: Path, stage: str, model_id: str,
                       audit: dict, cohort_ids: pd.Series, frozen_ids: pd.DataFrame,
                       expected_metrics: dict) -> tuple[pd.DataFrame, dict, dict]:
    root = work / '03_run/checkpoints' / stage / model_id
    manifest = jread(root / 'model_manifest.json')
    check(manifest.get('model_id') == model_id, f'Wrong model identifier in {model_id}')
    check(manifest.get('signature') == audit['signature'], f'Stale model signature: {model_id}')
    check(int(manifest.get('heldout_n', -1)) == EXPECTED_N, f'Wrong heldout n: {model_id}')
    check(int(manifest.get('development_n', -1)) == 4831, f'Wrong development n: {model_id}')
    check(len(manifest.get('features', [])) == 25, f'Not an SC 25-predictor model: {model_id}')
    check(all(p in manifest.get('output_sha256', {}) for p in
              ('final_model.cbm', 'heldout_predictions.csv', 'validation_history.json')),
          f'Missing checkpoint manifest hashes: {model_id}')
    for filename, checksum in manifest['output_sha256'].items():
        require_hash(root / filename, checksum)
    data = pd.read_csv(root / 'heldout_predictions.csv', dtype={'ResponseId': str})
    check(list(data.columns) == PREDICTION_FIELDS, f'Unexpected prediction fields for {model_id}: {list(data.columns)}')
    check(len(data) == EXPECTED_N, f'Incorrect row count in {model_id}')
    check(data['ResponseId'].notna().all() and data['ResponseId'].is_unique,
          f'Null/duplicate ResponseIds in {model_id}')
    idx = pd.to_numeric(data['canonical_row_index'], errors='raise').to_numpy()
    check(np.isfinite(idx).all() and np.all(idx == idx.astype(int)),
          f'Invalid canonical row indices in {model_id}')
    idx = idx.astype(int)
    check((idx >= 0).all() and (idx < len(cohort_ids)).all() and len(set(idx)) == EXPECTED_N,
          f'Out of range or duplicate index in {model_id}')
    check(np.array_equal(cohort_ids.iloc[idx].to_numpy(), data['ResponseId'].to_numpy()),
          f'Checkpoint rows do not match locked cohort ResponseIds: {model_id}')
    check(np.array_equal(frozen_ids['canonical_row_index'].to_numpy(), idx) and
          np.array_equal(frozen_ids['ResponseId'].to_numpy(), data['ResponseId'].to_numpy()),
          f'Checkpoint rows not aligned to frozen held-out partition: {model_id}')

    y = data['observed_WSQS_100'].to_numpy(dtype=float)
    p = data['predicted_WSQS_100'].to_numpy(dtype=float)
    res = data['residual_observed_minus_predicted'].to_numpy(dtype=float)
    check(np.isfinite(res).all() and np.allclose(y - p, res, rtol=0, atol=1e-10),
          f'Saved residuals fail prediction reconciliation: {model_id}')
    check((y >= 0).all() and (y <= 100).all(), f'Observed WSQS outside 0-100: {model_id}')
    computed = metrics_from_arrays(y, p)
    for key in ('rmse_100', 'mae_100', 'r2', 'spearman'):
        check(np.isclose(computed[key], manifest['metrics'][key], rtol=0, atol=5e-10),
              f'{model_id} {key} diverges from model checkpoint: {computed[key]} vs {manifest["metrics"][key]}')
        check(np.isclose(computed[key], expected_metrics[key], rtol=0, atol=5e-10),
              f'{model_id} {key} diverges from verified Stage04/05 aggregate metric')

    return data, computed, {
        'checkpoint': str(root.relative_to(work)),
        'prediction_sha256': sha(root / 'heldout_predictions.csv'),
        'model_sha256': sha(root / 'final_model.cbm'),
        'model_signature': manifest['signature'],
        'n': len(data),
        'metrics_verified': True,
    }


def save_new(path: Path, content: str) -> None:
    check(not path.exists(), f'Will not overwrite previous artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, suffix='.partial', delete=False) as stream:
        temp = Path(stream.name)
        stream.write(content)
    os.replace(temp, path)


def main(base: Path) -> None:
    os.umask(0o077)
    work = base.expanduser().resolve() / PROJECT
    expected_python = work / '.venv_wsqs_n6039/bin/python'
    check(Path(sys.executable).resolve() == expected_python.resolve(),
          'Please use the isolated N6039 Python environment')
    cohort_path = work / '00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    lock = jread(work / '07_audit/COHORT_DATASET_LOCK_N6039.json')
    require_hash(cohort_path, SOURCE_SHA)
    check(lock['output_sha256']['primary'] == SOURCE_SHA, 'Cohort lock has unexpected SHA')
    a04 = jread(work / '07_audit' / f'{STAGE04}_COMPLETE.json')
    a05 = jread(work / '07_audit' / f'{STAGE05}_COMPLETE.json')
    check(a04['status'] == a05['status'] == 'PASS', 'Prerequisite stage not PASS')
    check(a04['fingerprint']['source_cohort_sha256'] == SOURCE_SHA, 'Stage04 source hash mismatch')
    check(a05['inputs']['dataset_sha256'] == SOURCE_SHA and
          a05['inputs']['stage04_signature'] == a04['signature'],
          'Stage05 prerequisite chain not frozen to Stage04 source')

    p04 = work / '04_numerical_results/modeling' / STAGE04 / '01_primary_block_heldout_metrics.csv'
    p05 = work / '04_numerical_results/modeling' / STAGE05 / '01_original_eight_sensitivity_model_metrics.csv'
    require_hash(p04, a04['output_sha256'][p04.name])
    require_hash(p05, a05['output_sha256'][str(p05.relative_to(work))])
    m04, m05 = pd.read_csv(p04), pd.read_csv(p05)

    # The primary split is frozen in Stage04; use it instead of recreating a split.
    partition = work / '03_run/checkpoints' / STAGE04 / 'frozen_primary_partition_private.csv'
    check(partition.is_file(), 'Missing frozen primary partition')
    cohorts = pd.read_csv(cohort_path, usecols=['ResponseId'], dtype={'ResponseId': str})
    ids = cohorts['ResponseId']
    check(len(ids) == 6039 and ids.notna().all() and ids.is_unique,
          'Unexpected cohort ID count or duplicates')
    split = pd.read_csv(partition, dtype={'ResponseId': str})
    check({'canonical_row_index', 'ResponseId', 'partition'}.issubset(split.columns),
          'Malformed frozen partition file')
    check(len(split) == 6039 and split['canonical_row_index'].is_unique and
          set(split['canonical_row_index'].astype(int)) == set(range(6039)),
          'Frozen partition index malformed')
    split = split.sort_values('canonical_row_index')
    check(np.array_equal(split['ResponseId'].to_numpy(), ids.to_numpy()),
          'Frozen partition ResponseIds do not align with cohort')
    check((split.partition == 'heldout_test').sum() == EXPECTED_N, 'Frozen split must have 1208 heldout records')
    held = split[split.partition == 'heldout_test'][['canonical_row_index', 'ResponseId']]

    # Model checkpoint ordering is often shuffled; all 3 files must agree exactly.
    # Use row-index correspondence against the frozen split, then first checkpoint's
    # actual order as the canonical comparison order.
    official_idx = set(held['canonical_row_index'].to_numpy(dtype=int))
    baseline_ids = None
    rows, checkpoints = [], {}
    reference_features = None
    for name, stage, model_id, key in CONFIG:
        table = m04.loc[m04.block == key] if stage == STAGE04 else m05.loc[
            (m05.family == 'outcome') & (m05.specification == key)]
        check(len(table) == 1, f'Cannot identify one verified metric row: {model_id}')
        metrics = table.iloc[0].to_dict()
        # For index/order check, read saved index sequence once from checkpoint itself.
        raw = pd.read_csv(work / '03_run/checkpoints' / stage / model_id / 'heldout_predictions.csv',
                          usecols=['canonical_row_index', 'ResponseId'], dtype={'ResponseId': str})
        check(set(raw['canonical_row_index'].to_numpy(dtype=int)) == official_idx,
              f'Wrong heldout partition: {model_id}')
        if baseline_ids is None:
            baseline_ids = raw.copy()
        data, checked_metrics, provenance = inspect_checkpoint(work, stage, model_id,
                                                               a04 if stage == STAGE04 else a05,
                                                               ids, baseline_ids, metrics)
        features = jread(work / '03_run/checkpoints' / stage / model_id / 'model_manifest.json')['features']
        if reference_features is None:
            reference_features = features
        check(features == reference_features,
              f'Predictor set/order for {model_id} differs from frozen Stage04 SC')
        row = {
            'WSQS outcome': name,
            'Test N': len(data),
            'RMSE': checked_metrics['rmse_100'],
            'MAE': checked_metrics['mae_100'],
            'R2': checked_metrics['r2'],
            'Spearman rho': checked_metrics['spearman'],
            'Observed mean': float(data['observed_WSQS_100'].mean()),
            'Predicted mean': float(data['predicted_WSQS_100'].mean()),
        }
        rows.append(row)
        checkpoints[model_id] = provenance
        print(f'PASS | {model_id:>27} | N={len(data)} | observed mean={row["Observed mean"]:.9f} | '
              f'predicted mean={row["Predicted mean"]:.9f}', flush=True)

    # All observational cohorts should represent a shared set, but the three
    # outcomes are distinct. Disallow an accidentally mixed outcome definition.
    check(len(rows) == 3 and all(0 <= r['Observed mean'] <= 100 for r in rows),
          'Invalid Table08 means')
    check(abs(rows[0]['Observed mean'] - rows[1]['Observed mean']) < 5 and
          abs(rows[0]['Observed mean'] - rows[2]['Observed mean']) < 5,
          'Alternative outcome distributions diverge unexpectedly')

    outdir = work / '04_numerical_results/figure_sources' / OUTSTAGE
    auditpath = work / '07_audit' / f'{OUTSTAGE}_COMPLETE.json'
    if auditpath.exists():
        previous = jread(auditpath)
        check(previous['status'] == 'PASS' and previous['source_cohort_sha256'] == SOURCE_SHA,
              'Existing Stage16 audit is stale')
        for relpath, expected_hash in previous['output_sha256'].items():
            require_hash(work / relpath, expected_hash)
        check(previous['checkpoint_sources'] == checkpoints,
              'Saved Stage16 audit checkpoint signatures differ from current sources')
        print('PASS | Stage16 already complete; previous aggregate outputs and hashes verified')
        print('TABLE:', outdir / '02_TABLE08_publication_ready.md')
        return
    check(not outdir.exists(), 'Stage16 output folder exists without passing audit; refusing overwrite')
    outdir.mkdir(parents=True)
    values = pd.DataFrame(rows)
    csv_path = outdir / '01_TABLE08_reconciled_heldout_means_and_metrics.csv'
    csv_text = values.to_csv(index=False, float_format='%.17g')
    save_new(csv_path, csv_text)

    lines = [
        '# Table 8. Robustness across alternative WSQS outcome definitions',
        '',
        '| WSQS outcome | Test N | RMSE | MAE | R² | Spearman ρ | Observed mean | Predicted mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f'| {row["WSQS outcome"]} | {row["Test N"]} | {row["RMSE"]:.4f} | '
                     f'{row["MAE"]:.4f} | {row["R2"]:.6f} | {row["Spearman rho"]:.6f} | '
                     f'{row["Observed mean"]:.4f} | {row["Predicted mean"]:.4f} |')
    lines.extend(['',
        'Note. Same frozen held-out sample of 1,208 households for all three outcomes. '
        'Models contain 22 service-support and three household-context predictors. '
        'The population-priority outcome follows the original development-only weight construction; '
        'the individual-priority outcome is constructed from each household’s ratings. '
        'All eight numeric columns are recovered or verified from saved model checkpoints. '
        'Observed and predicted means are on the 0–100 WSQS scale. '
        'No model was trained or refitted in this recovery.', '',
    ])
    md_path = outdir / '02_TABLE08_publication_ready.md'
    save_new(md_path, '\n'.join(lines))
    audit = {
        'status': 'PASS', 'stage': OUTSTAGE, 'source_cohort_sha256': SOURCE_SHA,
        'stage04_signature': a04['signature'], 'stage05_signature': a05['signature'],
        'heldout_n': EXPECTED_N, 'development_n': 4831,
        'outcomes': ['CORE8 (SC)', 'POPULATION_PRIORITY (SC)', 'INDIVIDUAL_PRIORITY (SC)'],
        'same_heldout_indices_and_ResponseIds_for_all_three': True,
        'checkpoint_hashes_and_performance_metrics_reverified': True,
        'models_fitted': 0, 'models_refitted': 0, 'respondent_rows_exported': 0,
        'checkpoint_sources': checkpoints,
        'output_sha256': {
            str(p.relative_to(work)): sha(p) for p in (csv_path, md_path)
        },
        'code_sha256': sha(Path(__file__).resolve()),
    }
    save_new(auditpath, json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print('='*78)
    print('PASS | Table 8 six missing means recovered and verified')
    print('PASS | All 3 historical checkpoint hashes and recorded performance metrics matched')
    print('PASS | All 3 sets of 1208 heldout household identities and row indices matched')
    print('PASS | ZERO fits, ZERO respondent-level outputs')
    print('AGGREGATE CSV:', csv_path)
    print('PUBLICATION TABLE:', md_path)
    print('AUDIT:', auditpath)
    print('='*78)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path,
                        default=Path.home() / 'Desktop/Spring 2026/Chapter 3 MANUAL RERUN')
    args = parser.parse_args()
    try:
        main(args.base)
    except Exception as exc:
        print(f'FAILED | {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
