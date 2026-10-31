#!/usr/bin/env python3
"""WSQS Stage 11: read-only audit of numerical results and 56-exhibit dependency map.

No modeling, changes to upstream inputs, or edits to manuscript/figure files.
Output writes only to a new Stage11 audit folder; optional --check-only writes nothing.

Run:
 WORK="$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_WSQS_RERUN_20261030_N6039"
 "$WORK/.venv_wsqs_n6039/bin/python" ~/Downloads/WSQS_STAGE11_NUMERICAL_EXHIBIT_AUDIT_N6039.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = "NPJ_WSQS_RERUN_20261030_N6039"
SOURCE_SHA = "c72b8480d3cff223d1d4cf0ebcc29a3dd80204d563cd63a4295f6739f0ed77a1"
STAGES = {
    "04": ("modeling", "STAGE04_PRIMARY_CROSSDOMAIN_N6039"),
    "05": ("modeling", "STAGE05_SENSITIVITY_SATISFACTION_N6039"),
    "06": ("modeling", "STAGE06_REPEATED_SPLITS_BENCHMARK_N6039"),
    "07": ("inference", "STAGE07_PRIORITY_NULL_B1000_N6039"),
    "08": ("inference", "STAGE08_SHAP_ATTRIBUTION_N6039"),
    "09": ("inference", "STAGE09_SERVICE_RESPONSE_B200_N6039"),
    "10": ("measurement", "STAGE10_MEASUREMENT_PROVIDER_N6039"),
}
# Numerical provenance, not a statement that publication artwork/text is finished.
# SIDE = independent descriptive/geographic/temporal sidecar, not part of modeling runs.
# STATIC = verify previously locked questionnaire/method/reference figure or table.
EXHIBITS = [
    # id, description, numeric sources, external/static dependencies
    ("F1", "Study area / provision map", "10", "SIDE"),
    ("F2", "Eight service domains and priorities", "10", ""),
    ("F3", "WSQS score structure / weighting", "10,05", ""),
    ("F4", "Primary-provision service profiles", "10", ""),
    ("F5", "Nested and independent block prediction", "04,07", ""),
    ("F6", "SHAP + cross-domain attribution", "04,08", ""),
    ("F7", "Model-derived service-response curves", "09", ""),
    ("F8", "Repeated splits and learner robustness", "05,06", ""),
    ("T1", "CORE8 and eight domains", "10", ""),
    ("T2", "Stated priority summary", "10", ""),
    ("T3", "WSQS weighting sensitivity", "10,05", ""),
    ("T4", "Nested model held-out performance", "04", ""),
    ("T5", "Grouped SHAP and block Shapley R2", "04,08", ""),
    ("T6", "Top individual SHAP predictors", "08", ""),
    ("T7", "Priority-conditioned response contrasts", "09", ""),
    ("T8", "Alternative WSQS outcomes", "05", ""),
    ("SF1", "Survey response and exclusion flow", "COHORT", "SIDE"),
    ("SF2", "Survey participation timestamps", "", "SIDE"),
    ("SF3", "30-item exploratory factor structure", "10", ""),
    ("SF4", "WSQS weighting agreement", "10", ""),
    ("SF5", "Score coding transformation schematic", "", "STATIC"),
    ("SF6", "Full 33-variable SHAP ranking", "08", ""),
    ("SF7", "Priority modification uncertainty", "09", ""),
    ("SF8", "Household benchmark / provision composition", "10", "SIDE,STATIC"),
    ("SF9", "Geographic and settlement breadth", "", "SIDE"),
    ("ST1", "Analytical sample and partition construction", "COHORT,04", "SIDE"),
    ("ST2", "Karur survey-screen illustration", "", "SIDE"),
    ("ST3", "Demographics and education", "", "SIDE"),
    ("ST4", "CORE8 domain item/predictor exclusion map", "", "STATIC"),
    ("ST5", "Seven/eight domain completeness", "10", ""),
    ("ST6", "Questionbook, coding and survey response map", "", "STATIC"),
    ("ST7", "CORE8 original response frequencies", "10", ""),
    ("ST8", "Exploratory factor diagnostics", "10", ""),
    ("ST9", "CORE8 alpha and 95% intervals", "10", ""),
    ("ST10", "Thirty-item factor loadings", "10", ""),
    ("ST11", "Predictor block inclusion / roles", "", "STATIC"),
    ("ST12", "Support predictor definitions", "", "STATIC"),
    ("ST13", "Household context definitions", "", "STATIC"),
    ("ST14", "Municipal-use evidence and Q9 applicability", "10", ""),
    ("ST15", "Model design and validation denominators", "04,05,COHORT", "STATIC"),
    ("ST16", "Ten split block predictive performance", "06", ""),
    ("ST17", "Paired incremental priority changes", "06", ""),
    ("ST18", "Order independent block Shapley", "04", ""),
    ("ST19", "Seven learner benchmark", "06", ""),
    ("ST20", "Alternative predictors", "05", ""),
    ("ST21", "Priority-weighted alternative outcomes", "05", ""),
    ("ST22", "Eight domain cross-prediction + new CP baselines", "04", ""),
    ("ST23", "CORE7A/CORE7B sensitivity", "05,10", ""),
    ("ST24", "Learner hyperparameters / tree selection", "04,05", "STATIC"),
    ("ST25", "Partitions and reproducibility environment", "COHORT,04", "STATIC"),
    ("ST26", "Grouped SHAP / conditional bootstrap", "08", ""),
    ("ST27", "All 22 support–priority pair mappings", "09,10", "STATIC"),
    ("ST28", "Five-level predicted service responses", "09", ""),
    ("ST29", "Adjacent-category B200 intervals and counts", "09", ""),
    ("ST30", "External benchmark harmonization", "", "SIDE,STATIC"),
    ("ST31", "Geography/settlement subgroup evidence", "", "SIDE"),
    # New PI analyses are sources, but not counted among the original 56.
]
EXPECTED_OUTPUTS = {
    "04": ("01_primary_block_heldout_metrics.csv", "02_three_block_shapley_r2.csv", "03_cross_domain_full_vs_CP_heldout.csv"),
    "05": ("01_original_eight_sensitivity_model_metrics.csv", "02_WSQS_weighting_and_domain_exclusion_agreement.csv", "03_Q6_1_satisfaction_vs_WSQS_and_CORE7.csv", "04_Q6_1_fixed_satisfaction_CORE7A_distribution.csv", "05_Q6_1_fixed_satisfaction_domain_profiles.csv", "06_population_priority_weights_and_fallback.json"),
    "06": ("01_ten_split_all_block_metrics.csv", "02_ten_split_block_summary.csv", "03_ten_split_learner_benchmark_metrics.csv", "04_ten_split_learner_benchmark_summary.csv", "05_matched_split_incremental_priority_and_context.csv", "06_paired_rmse_r2_summary.csv", "07_matched_partition_manifest_no_ids.csv"),
    "07": ("01_priority_null_B1000_repetitions.csv", "02_priority_null_B1000_summary.csv", "03_priority_null_B1000_method.json"),
    "08": ("01_grouped_shap_with_bootstrap_ci.csv", "02_all_33_predictors_shap_with_bootstrap_ci.csv", "03_top15_shap_predictors.csv", "04_bootstrap_group_distribution_B1000.csv", "05_shap_additivity_and_sources.json"),
    "09": ("01_S28_response_profiles_with_B200_intervals.csv", "02_MAIN_TABLE07_priority_response_contrasts.csv", "03_S29_adjacent_severity_changes_with_B200_frequency.csv", "04_S7_priority_modification_low_high_B200.csv", "05_two_frozen_only_pairs_no_B200.csv", "06_service_response_eligibility_and_quartiles.csv", "07_B200_all_refit_scenario_means_PRIVATE.csv", "08_service_response_analysis_manifest.json"),
    "10": ("01_core8_measurement_summary.json", "02_domain_level_summary.csv", "03_core8_question_response_frequencies.csv", "04_priority_domains_and_weights.csv", "05_stated_priority_response_frequencies.csv", "06_weighting_and_core7_agreement.csv", "07_core8_alpha_and_bootstrap.json", "08_factor_structure_summary.json", "09_factor_loadings_30_items.csv", "10_factor_parallel_analysis_eigenvalues.csv", "11_factor_spearman_high_similarity_pairs.csv", "12_provider_evidence_summary.json", "13_provider_Q9_and_fitted_item_response_status.csv", "14_provider_by_primary_domestic_source.csv", "15_service_support_22_original_response_frequencies.csv", "16_weighting_household_scores_PRIVATE.csv", "17_primary_provision_domain_profiles.csv"),
}
EXPECTED_STAGE_N = {"04":3, "05":6, "06":7, "07":3, "08":5, "09":8, "10":17}


def req(cond, message):
    if not cond:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def source_base(stage, fname):
    if stage == '05' and fname != '01_original_eight_sensitivity_model_metrics.csv':
        return '04_numerical_results/measurement/' + STAGES[stage][1]
    return '04_numerical_results/' + STAGES[stage][0] + '/' + STAGES[stage][1]


def audit_hash_table(stage, audit):
    if stage == '06': return audit['aggregate_output_sha256']
    if stage == '09': return audit['result_sha256']
    return audit['output_sha256']


def inspect_one(work, stage):
    part, name = STAGES[stage]
    audit_path = work / '07_audit' / (name + '_COMPLETE.json')
    req(audit_path.is_file(), f'Missing completed audit for Stage{stage}: {audit_path}')
    report = read_json(audit_path)
    req(report.get('status') == 'PASS', 'Stage '+stage+' did not PASS')
    hashes = audit_hash_table(stage, report)
    req(isinstance(hashes,dict) and hashes, 'No output SHA ledger for Stage'+stage)
    expected = EXPECTED_OUTPUTS[stage]
    req(len(expected) == EXPECTED_STAGE_N[stage], f'Stage{stage} expectation ledger wrong')
    rows = []
    mapped = set()
    for filename in expected:
        rel = source_base(stage, filename) + '/' + filename
        p = work / rel
        req(p.is_file(), f'Missing Stage{stage} source: {rel}')
        # Some audit ledgers use root-relative filenames, others only basenames.
        observed = hashes.get(rel, hashes.get(filename))
        req(observed is not None, f'No hash in Stage{stage} audit for {rel}')
        req(sha(p) == observed, f'SHA mismatch for completed scientific output: {rel}')
        mapped.add(rel if rel in hashes else filename)
        rows.append({'stage':stage,'relative_path':rel,'bytes':p.stat().st_size,
                     'sha256':observed, 'privacy':'PRIVATE' if 'PRIVATE' in filename else 'AGGREGATE'})
    req(set(hashes) == mapped, 'Stage '+stage+' extra/unmapped outputs in immutable audit: '+str(set(hashes)-mapped))
    return report, rows


def check_numerical_invariants(work, audits):
    stage = lambda s, filename: work / source_base(s, filename) / filename
    m = read_json(stage('10', '01_core8_measurement_summary.json'))
    req(m['N']==6039, 'Stage10 wrong N')
    req(m['seven_domain_n']==266 and m['eight_domain_n']==5773, 'Wrong seven/eight domain counts')
    req(m['missing_by_domain']['Customer_service']==262 and m['missing_by_domain']['Quality']==4,
        'Domain availability mismatch against row-level reconciliation')
    req(sum(m['missing_by_domain'].values()) == 266, 'Seven-domain aggregate not explained by domain missingness')
    req(abs(m['CORE8_mean']-71.741566721)<5e-10, 'CORE8 mean changed')
    a=read_json(stage('10','07_core8_alpha_and_bootstrap.json'))
    req(abs(float(a['alpha_raw_complete_case'])-0.757709)<2e-6,'CORE8 alpha unexpected')
    factor=read_json(stage('10','08_factor_structure_summary.json'))
    req(factor['item_count']==30 and factor['factors_retained']==5 and factor['parallel_null_repetitions']==300,
        'Factor inventory/retention changed')
    req(abs(factor['overall_kmo']-0.941595)<2e-6,'KMO unexpected')
    provider=read_json(stage('10','12_provider_evidence_summary.json'))
    req(provider['N']==6039 and provider['municipal_positive_evidence']==4682,'Provider evidence count changed')
    model=pd.read_csv(stage('04','01_primary_block_heldout_metrics.csv'))
    req(len(model)==8, 'Expected mean baseline + seven block models')
    d={str(r.block):r for r in model.itertuples(index=False)}
    req(all(k in d for k in ['S','C','P','SC','SP','CP','SCP']), 'Incomplete block model set')
    req(abs(float(d['SCP'].r2)-0.548950096)<1e-8,'Stage04 SCP R2 changed')
    req(int(d['SCP'].heldout_n)==1208,'Stage04 heldout N changed')
    cross=pd.read_csv(stage('04','03_cross_domain_full_vs_CP_heldout.csv'))
    req(len(cross)==8 and cross['domain'].is_unique and
        (cross['incremental_other_domain_service_support_r2']>0).all(), 'Eight paired cross-domain results wrong')
    splits=pd.read_csv(stage('06','01_ten_split_all_block_metrics.csv'))
    req(len(splits)==80 and splits['seed'].nunique()==10, 'Stage06 expected 10 x 8 model rows')
    null=pd.read_csv(stage('07','01_priority_null_B1000_repetitions.csv'))
    summary=pd.read_csv(stage('07','02_priority_null_B1000_summary.csv')).iloc[0]
    req(len(null)==1000 and null['replicate_index'].is_unique,'Stage07 null B1000 record count wrong')
    exceed=int((null['null_delta_r2']>=summary['observed_incremental_r2']).sum())
    req(exceed==4 and int(summary['null_greater_or_equal_observed'])==4,
        'Stage07 priority-null exceedances differ')
    req(abs((exceed+1)/1001-summary['empirical_upper_tail_p_plus_one'])<1e-12,
        'Stage07 plus-one null p inconsistency')
    shap=pd.read_csv(stage('08','01_grouped_shap_with_bootstrap_ci.csv'))
    req(len(shap)==3 and abs(shap['percent_of_total_absolute_shap'].sum()-100)<1e-7,
        'Stage08 SHAP group shares incomplete')
    profiles=pd.read_csv(stage('09','01_S28_response_profiles_with_B200_intervals.csv'))
    req(len(profiles)==20,'Stage09 must have 4 x 5 response predictions')
    contrast=pd.read_csv(stage('09','02_MAIN_TABLE07_priority_response_contrasts.csv'))
    req(len(contrast)==4,'Stage09 Table7 four main contrasts missing')
    man=read_json(stage('09','08_service_response_analysis_manifest.json'))
    req(man['B']==200 and man['pressure_eligible_heldout_n']==943 and man['heldout_n']==1208,
        'Stage09 refitting or eligible N inconsistent')
    req(audits['06']['repeated_split_seeds']==list(range(20260714,20260724)),
        'Stage06 original seed series differs')
    return {
        'cohort_n':6039, 'heldout_n':1208, 'seven_domain_n':266,
        'Customer_service_missing':262,'Quality_missing':4,
        'CORE8_mean':float(m['CORE8_mean']),
        'CORE8_alpha':float(a['alpha_raw_complete_case']),
        'KMO':float(factor['overall_kmo']), 'factor_count':int(factor['factors_retained']),
        'positive_municipal_evidence_n':int(provider['municipal_positive_evidence']),
        'full_model_R2':float(d['SCP'].r2),
        'priority_null_exceedances':4,'priority_null_repetitions':1000,
        'priority_null_plus_one_p':float(summary['empirical_upper_tail_p_plus_one']),
        'service_profile_refits':200, 'pressure_eligible_heldout':943,
    }


def exhibit_catalog():
    req(len(EXHIBITS) == 56, f'Exhibit register has {len(EXHIBITS)}, not 56 items')
    ids = [r[0] for r in EXHIBITS]
    req(len(set(ids)) == 56, 'Duplicate exhibit ID')
    req(set(i for i in ids if i.startswith('F')) == {f'F{i}' for i in range(1,9)},'Main figure inventory mismatch')
    req(set(i for i in ids if i.startswith('T')) == {f'T{i}' for i in range(1,9)},'Main table inventory mismatch')
    req(set(i for i in ids if i.startswith('SF')) == {f'SF{i}' for i in range(1,10)},'Supplementary figure inventory mismatch')
    req(set(i for i in ids if i.startswith('ST')) == {f'ST{i}' for i in range(1,32)},'Supplementary table inventory mismatch')
    rows=[]
    for ident, title, sources, flags in EXHIBITS:
        stages=[x for x in sources.split(',') if x]
        outstanding=[x for x in flags.split(',') if x]
        for s in stages: req(s in STAGES or s=='COHORT','Unknown exhibit source '+s)
        req(all(z in ('SIDE','STATIC') for z in outstanding),'Unknown exhibit action')
        status = 'AWAITING_SIDECAR' if 'SIDE' in outstanding else ('STATIC_METHOD_REVIEW' if 'STATIC' in outstanding else 'NUMERIC_SOURCE_PRESENT')
        rows.append({'exhibit':ident,'description':title,'numeric_dependencies':sources,
                     'additional_work':flags,'status':status,
                     'layout_rendered':'NO', 'manuscript_text_reconciled':'NO'})
    return pd.DataFrame(rows)


def save_once(path, text):
    req(not path.exists(),'Refusing to overwrite Stage11 output: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=Path.home()/'Desktop/Spring 2026/Chapter 3 MANUAL RERUN'/PROJECT)
    parser.add_argument('--check-only',action='store_true',help='Read-only verification; no audit files')
    args=parser.parse_args()
    work=args.work.expanduser().resolve()
    source=work/'00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    frozen=work/'07_audit/COHORT_DATASET_LOCK_N6039.json'
    req(source.is_file() and frozen.is_file(),'Missing frozen N6039 dataset/lock')
    req(sha(source)==SOURCE_SHA==read_json(frozen)['output_sha256']['primary'],
        'Frozen primary cohort checksum mismatch')
    audits={}; rows=[]
    print('='*77)
    print('STAGE 11 | READ-ONLY N6039 NUMERICAL SOURCE + 56-EXHIBIT AUDIT')
    print('LOCKED SOURCE SHA256:',SOURCE_SHA)
    print('='*77,flush=True)
    for stage in STAGES:
        audits[stage],checked=inspect_one(work,stage)
        rows.extend(checked)
        print(f'PASS | Stage {stage}: {len(checked)} scientific outputs hash verified',flush=True)
    req(len(rows)==49,'Stage04–10 numeric file count differs from expected 49')
    metrics=check_numerical_invariants(work,audits)
    print('PASS | N6039/266 seven-domain count/262 CS + 4 quality reconciled')
    print('PASS | 10 split seeds, 1000 priority null, 33-feature SHAP, 200 refits, 30-item factors')
    exhibits=exhibit_catalog()
    counts=Counter(exhibits['status'])
    print(f'PASS | Original exhibits: 8 figures + 8 tables + 9 SI figures + 31 SI tables = {len(exhibits)}')
    for status in ('NUMERIC_SOURCE_PRESENT','AWAITING_SIDECAR','STATIC_METHOD_REVIEW'):
        print(f'{status}: {counts[status]}')
    print('CAUTION | Numeric source present != publication figure/table rendered or reconciled.')
    print('CAUTION | SIDE rows require your independent geographic/temporal/demographic analyses.')
    if args.check_only:
        print('PASS | Read-only audit complete; no files written')
        return
    result=work/'07_audit/STAGE11_NUMERICAL_EXHIBIT_AUDIT_N6039'
    req(not result.exists() or not any(result.iterdir()),
        'Stage11 audit folder already contains outputs. Use --check-only or move the old folder after independent review.')
    result.mkdir(parents=True,exist_ok=True)
    src_df=pd.DataFrame(rows).sort_values(['stage','relative_path'])
    report={'status':'PASS','audit_type':'numerical provenance and exhibit coverage only',
            'source_sha256':SOURCE_SHA,'stage_checks':list(STAGES),
            'verified_numeric_result_files':len(rows),
            'counts_by_exhibit_status':dict(counts),
            'metrics':metrics,'stage_audit_sha256':{
                k:sha(work/'07_audit'/(name+'_COMPLETE.json')) for k,(_,name) in STAGES.items()},
            'new_PI_analyses_not_in_original_56':['eight matched cross-domain CP baselines','Q6.1 satisfaction and CORE7A profiles'],
            'not_claimed':['final figures generated','final tables typeset','sidecar complete','reproducibility public release complete']}
    src_df.to_csv(result/'01_verified_source_file_ledger.csv',index=False,float_format='%.17g')
    exhibits.to_csv(result/'02_exhibit_56_dependency_register.csv',index=False)
    save_once(result/'03_numerical_audit_summary.json',json.dumps(report,indent=2,allow_nan=False)+'\n')
    pending=exhibits[exhibits.status!='NUMERIC_SOURCE_PRESENT']
    lines=['# WSQS N6039 exhibit-source handoff','','All Stage04–10 numerical outputs verified against their saved SHA-256 hashes.','This does not certify figure or manuscript assembly.','',
           f'**Numerical files checked:** {len(rows)}',
           f'**Original exhibits:** {len(exhibits)}',
           f'**Fully numerically sourced (not rendered):** {counts["NUMERIC_SOURCE_PRESENT"]}',
           f'**Await independent descriptive sidecar:** {counts["AWAITING_SIDECAR"]}',
           f'**Static methods/coding review only:** {counts["STATIC_METHOD_REVIEW"]}',
           '', '## Outstanding sidecar/static dependencies']
    for r in pending.itertuples(index=False):
        lines.append(f'- **{r.exhibit}** {r.description} | {r.status} | {r.additional_work}')
    lines+=['','## New analyses requiring placement in existing or new SI material',
            '- Eight matched cross-domain context-plus-priorities baselines (Stage04).',
            '- Q6.1 correlation, CORE7A comparison and fixed-satisfaction profiles (Stage05).',
            '', '## Scientific cautions',
            '- Stage10 municipal-use **positive evidence** is not the number of confirmed exclusive municipal users.',
            '- CORE8 alpha and factor retention are descriptive, not external validation.',
            '- Crossdomain, SHAP and randomized-priority results are predictive, not causal.',
            '- Figures/tables still require style-preserving regeneration and manuscript number-by-number reconciliation.',
            '']
    save_once(result/'04_handoff_and_pending_exhibits.md','\n'.join(lines))
    print('PASS | Stage11 report and 56-exhibit register saved; original files untouched')
    print('REPORT FOLDER:',result)
    print('NEXT | Integrate independent descriptive sidecar, then regenerate locked-layout exhibits')
    print('='*77)

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print('FAILED |',type(e).__name__,str(e),file=sys.stderr,flush=True)
        raise
