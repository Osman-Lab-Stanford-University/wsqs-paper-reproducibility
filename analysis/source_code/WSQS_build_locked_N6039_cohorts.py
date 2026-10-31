#!/usr/bin/env python3
"""WSQS npj Clean Water | Build frozen N=6039 and N=6074 analytic CSVs.

Run on the researcher's Mac, from any working directory:
    python3 ~/Downloads/WSQS_build_locked_N6039_cohorts.py

This reads the historical Step09N source and exact frozen ResponseId exclusion CSVs.
It does NOT infer exclusions from timestamps or demographic values, change any
scientific scoring/model configuration, or modify the historical source.

PI analysis scope recorded for subsequent steps (NOT EXECUTED HERE):
- 1000+ priority-null permutations, original null/model procedure
- 8 matched context+priority-only cross-domain CatBoost baselines
- Q6.1 vs CORE-8/CORE-7A satisfaction diagnostics
- short provider Q1.1-Q1.3 vs Q9.1/Q9.2 applicability audit
- no new known-groups validity test; reply explains scientific boundary
- preserve 33 predictor blocks, CORE-8 scoring, original analysis methods
- all full-precision numerical sources for main and supplementary displays
- timestamps, state/PIN/education/external benchmarks may be separate sidecar
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BASE_REL = Path('Desktop/Spring 2026/Chapter 3 MANUAL RERUN')
PROJECT = 'NPJ_WSQS_RERUN_20261030_N6039'
SOURCE_REL = Path('NPJ_Revision_2026/04_results/step09N_provider_eligibility/20261009_002640_637282/chapter3_STEP09N_APPLICABILITY_ENRICHED_MASTER.csv')
EXCLUSION_REL = Path('WSQS_MODEL_RERUN_EXCLUSION_IDS')
EXPECTED_MASTER_SHA256 = '9b1224ffd80397cf919c48cd944b7e7e3e17c2dded8dd99cab77ef4894b7ce9a'
EXPECTED_N, EXPECTED_COLS = 6145, 182
FILES = {
    'primary': ('WSQS_EXCLUDE_106_PRIMARY_N6039.csv', 106),
    'temporal': ('WSQS_EXCLUDE_71_REFERENCE_N6074.csv', 71),
    'qa': ('WSQS_EXCLUDE_35_QA_SENSITIVITY.csv', 35),
    'manifest': ('WSQS_EXCLUSION_MANIFEST_106.csv', 106),
}
AUDIT_FILE = 'WSQS_EXCLUSION_FREEZE_AUDIT.json'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest_ordered(ids: list[str]) -> str:
    return hashlib.sha256(('\n'.join(ids) + '\n').encode('utf-8')).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as e:
            raise ValueError(f'Empty CSV: {path}') from e
        rows = list(reader)
    require(bool(header), f'CSV lacks header: {path}')
    require(len(header) == len(set(header)), f'Duplicate column headers: {path}')
    bad = [i for i, r in enumerate(rows, 2) if len(r) != len(header)]
    require(not bad, f'Unequal column count in {path.name}, CSV lines: {bad[:5]}')
    return header, rows


def response_col(header: list[str], path: Path) -> int:
    norm = lambda x: ''.join(ch for ch in x.lower() if ch.isalnum())
    matches = [i for i, c in enumerate(header) if norm(c) in ('responseid', 'qualtricsresponseid')]
    require(len(matches) == 1, f'Expected one ResponseId column in {path.name}, found {len(matches)}; columns: {header[:15]}')
    return matches[0]


def read_id_list(path: Path, expected_n: int) -> list[str]:
    header, rows = read_csv(path)
    j = response_col(header, path)
    ids = [r[j].strip() for r in rows]
    require(len(ids) == expected_n, f'{path.name}: expected {expected_n} records, found {len(ids)}')
    require(all(ids), f'{path.name}: at least one blank ResponseId')
    require(len(set(ids)) == len(ids), f'{path.name}: repeated ResponseIds')
    return ids


def write_csv_new(path: Path, header: list[str], rows: list[list[str]]) -> None:
    require(not path.exists(), f'Refusing to overwrite existing dataset: {path}')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix='.wsqs_cohort_', dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(header)
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def copy_frozen(source: Path, target: Path) -> str:
    original_hash = sha256(source)
    if target.exists():
        require(target.is_file() and sha256(target) == original_hash,
                f'Existing frozen input conflicts: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(source, target)
        target.chmod(0o600)
        require(sha256(target) == original_hash, f'Copy verification failed: {target}')
    return original_hash


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path.home() / BASE_REL,
                        help='Parent folder containing historical NPJ_Revision_2026 and exclusion IDs')
    parser.add_argument('--source', type=Path, default=None,
                        help='Override source CSV path; historical SHA-256 is STILL enforced')
    parser.add_argument('--expected-master-sha256', default=EXPECTED_MASTER_SHA256,
                        help=argparse.SUPPRESS)  # Internal use for synthetic testing only
    args = parser.parse_args()
    base = args.base.expanduser().resolve()
    master = (args.source.expanduser().resolve() if args.source is not None
              else base / SOURCE_REL)
    exclusion_dir = base / EXCLUSION_REL
    project = base / PROJECT

    print('=' * 76)
    print('WSQS | CONSTRUCT FROZEN N6039 PRIMARY / N6074 TEMPORAL-REFERENCE DATASETS')
    print('=' * 76, flush=True)

    require(master.is_file(), f'Historical master not found: {master}')
    require(exclusion_dir.is_dir(), f'Frozen exclusion folder not found: {exclusion_dir}')
    sources = {key: exclusion_dir / f for key, (f, _) in FILES.items()}
    sources['audit'] = exclusion_dir / AUDIT_FILE
    for key, f in sources.items():
        require(f.is_file(), f'Missing exclusion input {key}: {f}')

    # Validate inputs fully BEFORE writing data into the new project.
    master_hash = sha256(master)
    require(master_hash == args.expected_master_sha256,
            f'Master SHA-256 differs from historical Run 01 source; do not proceed.\n'
            f'Expected {args.expected_master_sha256}\nActual   {master_hash}\nFile: {master}')
    header, rows = read_csv(master)
    require(len(rows) == EXPECTED_N,
            f'Original master expected {EXPECTED_N} household rows; found {len(rows)}')
    require(len(header) == EXPECTED_COLS,
            f'Original master expected {EXPECTED_COLS} columns; found {len(header)}')
    j = response_col(header, master)
    all_ids = [r[j].strip() for r in rows]
    require(all(all_ids), 'Missing ResponseIds in historical master')
    require(len(set(all_ids)) == EXPECTED_N, 'Duplicate ResponseIds in historical master')
    master_ids = set(all_ids)
    excl = {k: read_id_list(sources[k], n) for k, (_, n) in FILES.items()}
    primary, temporal, qa = map(lambda k: set(excl[k]), ('primary', 'temporal', 'qa'))
    require(not (temporal & qa), 'Temporal and QA exclusion lists are NOT disjoint')
    require(primary == temporal | qa, 'Primary exclusions must equal 71 temporal + 35 QA IDs')
    require(set(excl['manifest']) == primary, '106-ID explanatory manifest does not match primary exclusions')
    for key, seq in excl.items():
        unknown = set(seq) - master_ids
        require(not unknown, f'{key} contains {len(unknown)} ResponseIds absent from historical master')

    primary_rows = [r for r in rows if r[j].strip() not in primary]
    temporal_rows = [r for r in rows if r[j].strip() not in temporal]
    require(len(primary_rows) == 6039 and len(temporal_rows) == 6074,
            f'Unexpected cohort sizes: primary {len(primary_rows)}, temporal {len(temporal_rows)}')
    p_ids = [r[j].strip() for r in primary_rows]
    t_ids = [r[j].strip() for r in temporal_rows]
    require(set(t_ids) - set(p_ids) == qa, 'Temporal-only minus primary must be exactly the 35 QA IDs')
    require(set(p_ids).isdisjoint(primary), 'Excluded ID remained in primary')
    require(set(t_ids).isdisjoint(temporal), 'Temporal-excluded ID remained in reference')

    print('PASS | Old master checksum matches Run 01')
    print(f'PASS | Original {len(rows)} households, {len(header)} columns, unique ResponseIds')
    print('PASS | Exactly 106 primary exclusion IDs = 71 temporal + 35 QA')
    print('PASS | All IDs match master, source manifest matches, groups disjoint')
    print('PASS | Predicted new cohorts: primary N=6039, temporal reference N=6074')

    output = project / '00_private_inputs/cohort'
    audit_dir = project / '07_audit'
    code_dir = project / '02_code_private/reconciled'
    source_dir = project / '00_private_inputs/source'
    outputs = {
        'primary': output / 'WSQS_PRIMARY_ANALYTICAL_N6039.csv',
        'reference': output / 'WSQS_TEMPORAL_REFERENCE_N6074.csv',
        'ids_primary': output / 'WSQS_PRIMARY_N6039_ResponseIds.csv',
        'ids_reference': output / 'WSQS_REFERENCE_N6074_ResponseIds.csv',
        'membership': output / 'WSQS_COHORT_MEMBERSHIP_PRIVATE.csv',
        'audit': audit_dir / 'COHORT_DATASET_LOCK_N6039.json',
    }
    # No partial replacements of previously constructed datasets.
    collisions = [str(f) for f in outputs.values() if f.exists()]
    require(not collisions,
            'Output files already exist; verify previous run instead of overwriting:\n' + '\n'.join(collisions))

    for d in (output, audit_dir, code_dir, source_dir):
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
    originals = {sources[k].name: copy_frozen(sources[k], output / sources[k].name)
                 for k in sources}
    original_master_frozen = source_dir / 'chapter3_STEP09N_APPLICABILITY_ENRICHED_MASTER_ORIGINAL_N6145.csv'
    copy_frozen(master, original_master_frozen)

    write_csv_new(outputs['primary'], header, primary_rows)
    write_csv_new(outputs['reference'], header, temporal_rows)
    write_csv_new(outputs['ids_primary'], ['ResponseId'], [[x] for x in p_ids])
    write_csv_new(outputs['ids_reference'], ['ResponseId'], [[x] for x in t_ids])
    membership_rows = [
        [rid, '1' if rid not in primary else '0', '1' if rid not in temporal else '0',
         'TEMPORAL' if rid in temporal else ('QA' if rid in qa else 'RETAIN')]
        for rid in all_ids
    ]
    write_csv_new(outputs['membership'],
                  ['ResponseId', 'primary_include', 'temporal_reference_include', 'selection_status'],
                  membership_rows)

    output_hashes = {name: sha256(f) for name, f in outputs.items() if name != 'audit'}
    run_audit = {
        'status': 'COHORT_DATASETS_CONSTRUCTED_AND_VERIFIED',
        'executed_utc': datetime.now(timezone.utc).isoformat(),
        'project': str(project),
        'master_source': str(master),
        'master_sha256': master_hash,
        'frozen_master_sha256': sha256(original_master_frozen),
        'master_records': EXPECTED_N,
        'master_column_count': len(header),
        'master_response_id_column': header[j],
        'primary_exclusions': 106,
        'temporal_exclusions': 71,
        'additional_qa_exclusions': 35,
        'primary_n': len(primary_rows),
        'temporal_reference_n': len(temporal_rows),
        'all_exclusion_ids_matched': True,
        'exclusion_manifest_matches': True,
        'temporal_qa_disjoint': True,
        'primary_ordered_response_ids_sha256': digest_ordered(p_ids),
        'reference_ordered_response_ids_sha256': digest_ordered(t_ids),
        'output_sha256': output_hashes,
        'frozen_exclusion_file_sha256': originals,
        'historical_sources_modified': False,
        'empirical_models_fitted': 0,
        'respondent_data_publicly_exported': False,
        'note': 'Only a private cohort build. Use this frozen primary CSV as input to all N6039 modeling. Descriptive sidecar must match its ResponseId hash.',
    }
    outputs['audit'].write_text(json.dumps(run_audit, indent=2) + '\n', encoding='utf-8')
    outputs['audit'].chmod(0o600)
    here = Path(__file__).resolve()
    saved_script = code_dir / here.name
    if saved_script != here:
        copy_frozen(here, saved_script)

    # Read back the artifacts to verify actual written rows and ordered IDs.
    for kind, expected, ordered in [('primary', 6039, p_ids), ('reference', 6074, t_ids)]:
        got_header, got_rows = read_csv(outputs[kind])
        require(got_header == header and len(got_rows) == expected,
                f'Output round-trip failed: {kind}')
        require([r[j].strip() for r in got_rows] == ordered,
                f'Output ordered ResponseIds differ: {kind}')
    print('PASS | Primary N=6039 dataset written and reread')
    print('PASS | Temporal-reference N=6074 dataset written and reread')
    print('PASS | Original 182 columns and canonical record order retained')
    print('PASS | Source, exclusion CSVs, code and output checksums saved')
    print('PASS | Historical input unchanged; no empirical model fitted')
    print('PROJECT:', project)
    print('PRIMARY_DATA:', outputs['primary'])
    print('REFERENCE_DATA:', outputs['reference'])
    print('COHORT_AUDIT:', outputs['audit'])
    print('=' * 76)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, csv.Error) as exc:
        print('FAILED | ' + str(exc), file=sys.stderr)
        sys.exit(1)
