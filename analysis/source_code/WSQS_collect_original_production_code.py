#!/usr/bin/env python3
"""Read-only source-code collector for the WSQS 2026 private production run.

Creates a NEW archive of code (and limited config text) only. It does NOT bundle
CSV/Excel/surveys/respondents/model artifacts/Word/PDF/figures/logs, and it does
NOT modify anything under the research directory. Review the archive before
sharing, as source code can contain embedded secrets or private paths.

Example:
 python3 WSQS_collect_original_production_code.py \
   --root "$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_Revision_2026"
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

DEFAULT_ROOT = Path.home() / 'Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_Revision_2026'
SCAN = (
    '02_code',
    '03_audits/reproducibility_package_build',
    '04_results',
)
EXTENSIONS = {'.py', '.sh', '.toml', '.yaml', '.yml'}
MAX_FILE_BYTES = 2_500_000
MAX_TOTAL_BYTES = 40_000_000
EXPECTED = [
    '02_code/STEP18B_corrected_pipeline_scaffold.py',
    '02_code/step09J_final_scoring_resolution.py',
    '02_code/step09K_final_scoring_and_model_preparation.py',
    '04_results/step19P5_controlled_research_production/NPJ_WSQS_20261013_RUN01',
    '04_results/FIGURES_FOR_MANUSCRIPT_20261014_000258_892',
    '04_results/step19P5_controlled_research_production/NPJ_WSQS_20261013_FIG06_TABLE07_REFIT_BOOTSTRAP_B200',
]
SKIP_PARTS = {'__pycache__', '.git', '.venv', 'venv', 'node_modules'}


def sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def safe_include(relative: Path) -> bool:
    if any(p in SKIP_PARTS or p.startswith('.') for p in relative.parts):
        return False
    suffix = relative.suffix.lower()
    if suffix in EXTENSIONS:
        return True
    # Source configuration only. No arbitrary JSON files (they may contain data).
    if suffix == '.json' and relative.parts[0] == '02_code':
        name = relative.stem.lower()
        return any(t in name for t in ('config', 'setting', 'schema', 'spec', 'plan'))
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--output', type=Path,
                   default=Path.home() / 'Desktop/WSQS_CODE_CONTEXT_ONLY.zip')
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f'ERROR: source project root not found: {root}')
    if output.exists():
        sys.exit(f'ERROR: output already exists and will not be overwritten: {output}')
    if output == root or root in output.parents:
        sys.exit('ERROR: output must be outside the protected project directory')

    chosen = []
    bytes_total = 0
    skipped = []
    for dirname in SCAN:
        directory = root / dirname
        if not directory.is_dir():
            skipped.append({'path': dirname, 'reason': 'missing_directory'})
            continue
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if not safe_include(relative):
                continue
            size = path.stat().st_size
            if size > MAX_FILE_BYTES or bytes_total + size > MAX_TOTAL_BYTES:
                skipped.append({'path': str(relative), 'reason': 'size_cap'})
                continue
            contents = path.read_bytes()
            if b'\x00' in contents:
                skipped.append({'path': str(relative), 'reason': 'binary_content'})
                continue
            try:
                contents.decode('utf-8')
            except UnicodeDecodeError:
                skipped.append({'path': str(relative), 'reason': 'non_utf8'})
                continue
            chosen.append((relative, contents))
            bytes_total += len(contents)

    if not chosen:
        sys.exit('ERROR: no safe source-code files found under the specified project root')
    source_index = [
        {'path': str(rel), 'bytes': len(data), 'sha256': sha256(data)}
        for rel, data in chosen
    ]
    report = {
        'purpose': 'Source and configurations only; NOT a data or result export',
        'project_root': str(root),
        'code_file_count': len(chosen),
        'code_total_bytes': bytes_total,
        'expected_paths': [
            {'path': name, 'exists': (root / name).exists()}
            for name in EXPECTED
        ],
        'included': source_index,
        'skipped': skipped,
        'warning': 'Source code might contain embedded private information. Review ZIP before sharing.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for relative, data in chosen:
            z.writestr('PROJECT_SOURCE/' + relative.as_posix(), data)
        z.writestr('SOURCE_INVENTORY.json', json.dumps(report, indent=2) + '\n')
    with zipfile.ZipFile(output) as z:
        broken = z.testzip()
        if broken:
            sys.exit(f'Archive CRC failure: {broken}')
    print('PASS | Source code collected (no CSV, respondent data, model or figure assets)')
    print('Code files:', len(chosen), '| Uncompressed bytes:', bytes_total)
    print('Output:', output)
    print('Missing expected paths:')
    for e in report['expected_paths']:
        if not e['exists']:
            print('  -', e['path'])
    print('WARNING | Review the source ZIP for embedded secrets/private information before sharing.')


if __name__ == '__main__':
    main()
