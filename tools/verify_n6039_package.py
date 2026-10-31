#!/usr/bin/env python3
"""Verify N6039 repository files, original code integrity, sources and tables."""
from pathlib import Path
import hashlib,json,sys,zipfile
R=Path(__file__).resolve().parent.parent
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((R/'analysis/audits/N6039_PACKAGE_MANIFEST_SHA256.json').read_text())
for e in manifest['files']:
    p=R/e['path']
    assert p.is_file() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],e['path']
print(f"PASS | {len(manifest['files'])} checksum-matched repository files")
assert (R/'tests/test_core7_predictors.py').exists()
assert (R/'src/wsqs/extended.py').exists()
assert 'CORE7B' in (R/'src/wsqs/extended.py').read_text()
assert (R/'publication/tables/MAIN_TABLES_1_TO_8_N6039.md').read_text().count('## Table ')==8
# The original CORE-7B full-block implementation and its test are part of
# the verified GitHub baseline and must not be silently replaced.
original_scientific_hashes = {
    'src/wsqs/extended.py': '8fb703666407d6520e2ec497081e519491e157bf7d280036555ff464d916b6d7',
    'tests/test_core7_predictors.py': '29e9a35c1675462030042a07d39661f241ab594deb2ebd59eaf344a40aba18b3',
}
for name, expected in original_scientific_hashes.items():
    assert sha(R/name) == expected, f'Original GitHub scientific source changed: {name}'
print('PASS | Existing CORE7B implementation, test and eight main tables')
G=json.loads((R/'analysis/audits/verified_combined_handoff_manifest.json').read_text())
for e in G['members']:
    n=e['name']
    if n.startswith(('numeric_sources/','figure_sources/')):
        p=R/'analysis/results'/Path(*Path(n).parts[1:])
        assert p.is_file() and sha(p)==e['sha256'],n
assert sum(p.suffix.lower() in {'.csv','.json'} for p in (R/'analysis/results').rglob('*') if p.is_file())==55
print('PASS | 55 N6039 numerical and figure-source aggregates')
C=json.loads((R/'coverage_analysis/audits/FILE_MANIFEST_SHA256.json').read_text())
for e in C:
    p=R/e['path']
    assert p.is_file() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],e['path']
assert len(list((R/'coverage_analysis/results').glob('*.csv')))==13
print('PASS | S4 source hashes and 13 aggregate results')
for p in R.rglob('*'):
    if p.is_file():
        assert p.suffix.lower() not in {'.cbm','.pkl','.parquet','.feather'},p
        assert 'PRIVATE' not in p.name and 'ResponseId' not in p.name,p
print('PASS | No respondent-level data or fitted model binaries')
print('PASS | WSQS N6039 repository integrity')
