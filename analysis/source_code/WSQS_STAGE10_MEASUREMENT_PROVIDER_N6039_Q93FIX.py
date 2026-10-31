#!/usr/bin/env python3
"""WSQS Stage10 | Full-cohort descriptive measurement and provider diagnostics.

Only reads the frozen 6039-household CSV, existing Stage03/05/09 audit files,
verified scoring and feature code; fits NO CatBoost models.

Outputs exact-precision numerical source tables supporting manuscript Figures 2-4,
Tables 1-3, SI S4-S10, S14, S27 and the targeted PI provider audit.

Factor diagnostics explicitly use 30 item-level indicators: the eight CORE8
items + 21 of 22 service-support items (exclude nominal Q2.8) + Q9.3. Q6.5
is not included. Exploratory median imputation affects factor analysis only.
A 30-item factor fit requires factor_analyzer==0.5.1 in the isolated venv:
  .venv_wsqs_n6039/bin/python -m pip install 'factor_analyzer==0.5.1'

Run with:
  WORK="$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_WSQS_RERUN_20261030_N6039"
  "$WORK/.venv_wsqs_n6039/bin/python" ~/Downloads/WSQS_STAGE10_MEASUREMENT_PROVIDER_N6039.py

If dependency unavailable, fail early before writing any outputs.
Source and historical files are read-only. All outputs write-once and SHA-verified.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = 'NPJ_WSQS_RERUN_20261030_N6039'
STAGE = 'STAGE10_MEASUREMENT_PROVIDER_N6039'
SOURCE_SHA = 'c72b8480d3cff223d1d4cf0ebcc29a3dd80204d563cd63a4295f6739f0ed77a1'
N, B_ALPHA, B_PARALLEL = 6039, 1000, 300
SEED_ALPHA, SEED_PARALLEL = 20260714, 20260715


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write_json(path, val):
    path = Path(path)
    require(not path.exists(), f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.part', dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        json.dump(val, f, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(temp, path)


def write_csv(path, df):
    path = Path(path)
    require(not path.exists(), f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.part', dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        df.to_csv(f, index=False, float_format='%.17g')
    os.replace(temp, path)


def alpha(matrix):
    x = np.asarray(matrix, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] >= 2 and np.isfinite(x).all(), 'Invalid alpha matrix')
    k = x.shape[1]
    variance = x.var(axis=0, ddof=1).sum()
    total = x.sum(axis=1).var(ddof=1)
    return float(k/(k-1) * (1 - variance/total)) if total > 0 else float('nan')


def stan_alpha(matrix):
    correlations = np.corrcoef(np.asarray(matrix, dtype=float), rowvar=False)
    k = correlations.shape[0]
    off = correlations[np.triu_indices(k, 1)]
    mean_r = float(off.mean())
    return float(k * mean_r / (1 + (k-1) * mean_r)), mean_r


def bootstrap_alpha(x, b, seed):
    rng = np.random.default_rng(seed)
    output = np.empty(b, dtype=float)
    for i in range(b):
        sample = rng.integers(0, len(x), len(x))
        output[i] = alpha(x[sample, :])
    require(np.isfinite(output).all(), 'Nonfinite bootstrap alpha')
    return [float(z) for z in np.quantile(output, [0.025,0.975])]


def q93_from_source(df):
    """Use harmonized numeric Q9.3 if present, otherwise strictly map the survey wording."""
    target = [c for c in df.columns if re.search(r'(?i)Q[ _]*9[._ ]3', c)]
    numeric = [c for c in target if 'severity' in c.lower()]
    if len(numeric) == 1:
        values = pd.to_numeric(df[numeric[0]], errors='coerce')
        require(values.dropna().isin([1,2,3,4,5]).all(), 'Q9.3 harmonized severity invalid')
        return values, numeric[0], 'source numeric severities'
    raw = [c for c in target if c.lower().endswith('raw_value')]
    require(len(raw) == 1, 'Cannot uniquely locate Q9.3 original response: '+str(target))
    raw_s = df[raw[0]].astype('string').str.strip().str.lower()
    expected = {
        'very easy':1, 'easy':2, 'neutral':3,
        'difficult':4, 'very difficult':5,
    }
    mapped = raw_s.map(expected).astype(float)
    unknown = sorted(set(raw_s.dropna().unique()) - set(expected))
    # Explicitly non-applicable/no channel answer is not a severity value.
    permit = {'no known communication channel', 'there is no known communication channel',
              'there is no stream of communication', 'no known stream of communication', '', 'not applicable',
              'no communication channel', 'not aware'}
    require(set(unknown).issubset(permit), 'Unclassified Q9.3 original responses: '+str(unknown))
    return mapped, raw[0], 'original response categories, with no channel unscored'


def category_counts(df, name, source, scope='all_households'):
    vals = df[source].astype('string').str.strip().fillna('<MISSING>')
    vals = vals.replace('', '<BLANK>')
    counts = vals.value_counts(dropna=False, sort=False)
    return [{'scope': scope, 'source': name, 'source_column': source,
             'response': str(val), 'n': int(count), 'fraction': float(count/len(df))}
            for val, count in counts.items()]


def make_factor(df, spec, raw, research_dir):
    from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo

    items = []
    for j, domain in enumerate(raw.columns):
        items.append((domain, domain+' CORE8', raw[domain]))
    for domain, mapping in spec['domains'].items():
        for qid, col in mapping.items():
            if qid == 'Q2.8':
                continue
            require(col in df, 'Missing factor input for '+qid+': '+col)
            items.append((domain, qid, pd.to_numeric(df[col], errors='coerce')))
    contact, contact_source, contact_policy = q93_from_source(df)
    items.append(('Customer_service', 'Q9.3', contact))
    require(len(items) == 30, f'Factor inventory must have 30 indicators; got {len(items)}')
    names = [name for _domain,name,_series in items]
    require(len(set(names)) == len(names), 'Duplicate factor diagnostic item')
    raw_matrix = pd.DataFrame({name: value.reset_index(drop=True) for domain,name,value in items})
    # Do not assign 'poor' values to missing or inapplicable survey responses.
    na_counts = raw_matrix.isna().sum()
    missing_pct = raw_matrix.isna().mean()
    require(raw_matrix.notna().sum().ge(3).all(), 'Empty factor diagnostic item')
    imputed = raw_matrix.apply(lambda s: s.fillna(s.median()), axis=0)
    require(np.isfinite(imputed.to_numpy(dtype=float)).all(), 'Nonfinite factor matrix')
    nonzero = imputed.std(ddof=1).gt(0)
    require(nonzero.all(), 'Zero-variance factor items: '+str(nonzero.index[~nonzero].tolist()))
    R = imputed.corr().to_numpy(dtype=float)
    require(np.linalg.eigvalsh(R).min() > 0, 'Singular imputed factor correlation matrix')
    kmo_item, overall_kmo = calculate_kmo(imputed)
    bart_chi2, bart_p = calculate_bartlett_sphericity(imputed)
    obs = np.linalg.eigvalsh(R)[::-1]
    X = imputed.to_numpy(dtype=float)
    rng = np.random.default_rng(SEED_PARALLEL)
    simulated = np.empty((B_PARALLEL, len(names)), dtype=float)
    for b in range(B_PARALLEL):
        shuffled = np.empty_like(X)
        for c in range(X.shape[1]):
            shuffled[:,c] = X[rng.permutation(len(X)),c]
        simulated[b] = np.linalg.eigvalsh(np.corrcoef(shuffled, rowvar=False))[::-1]
    null95 = np.quantile(simulated, .95, axis=0)
    keep = int((obs > null95).sum())
    require(1 <= keep < 15, 'Unexpected parallel-analysis factor count: '+str(keep))
    fa = FactorAnalyzer(n_factors=keep, method='minres', rotation='oblimin')
    fa.fit(imputed)
    loadings = np.asarray(fa.loadings_, dtype=float)
    communality = np.asarray(fa.get_communalities(), dtype=float)
    require(loadings.shape == (30, keep), 'Factor loading shape invalid')
    # Mark primary factor by largest abs loading; factor signs are arbitrary.
    main = np.argmax(np.abs(loadings), axis=1)
    factor_rows=[]
    for i,(domain, name, series) in enumerate(items):
        a = np.abs(loadings[i])
        sorted_a = np.sort(a)[::-1]
        salient = np.flatnonzero(a >= .3)
        item={'domain':domain, 'item':name, 'n_observed':int(series.notna().sum()),
              'n_imputed_median':int(series.isna().sum()), 'primary_factor':f'F{main[i]+1}',
              'primary_pattern_loading':float(loadings[i,main[i]]),
              'communality':float(communality[i]),
              'salient_loading_count':int(len(salient)),
              'weak_primary_loading':bool(a.max() < .3),
              'cross_loading_at_0_3':bool(len(salient)>=2)}
        item.update({f'F{k+1}_pattern':float(loadings[i,k]) for k in range(keep)})
        factor_rows.append(item)
    # Observe rank redundancy on genuinely paired survey values, not factor-imputed ones.
    sp = raw_matrix.corr(method='spearman').to_numpy(dtype=float)
    tri = np.triu_indices(30, 1)
    rho = sp[tri]
    rows=[]
    for i,j,r in zip(tri[0],tri[1],rho):
        if np.isfinite(r) and abs(r)>=.8:
            rows.append({'item_A':names[i], 'item_B':names[j], 'spearman_rho':float(r),
                         'threshold_0_8':True,'threshold_0_9':abs(r)>=.9})
    payload={
        'item_count':len(names),'n_households':N,
        'q93_source_column':contact_source,'q93_treatment':contact_policy,
        'excluded_from_factor':['Q2.8 (nonordinal categorical)', 'Q6.5 (not part of the 30-item diagnostic set)'],
        'factor_input_missingness':{name:{'n_missing':int(na_counts[name]),
                                             'missing_fraction':float(missing_pct[name])} for name in names},
        'factor_imputation':'median, exploratory diagnostics only, never WSQS scoring',
        'parallel_null_repetitions':B_PARALLEL,
        'parallel_seed':SEED_PARALLEL,
        'parallel_reference':'columnwise independently permuted 95th percentile',
        'factors_retained':keep,'overall_kmo':float(overall_kmo),
        'bartlett_chisquare':float(bart_chi2),'bartlett_df':int(len(names)*(len(names)-1)/2),
        'bartlett_p':float(bart_p),
        'alpha_30_imputed':alpha(X),
        'high_similarity_pairs_ge_0_8':len(rows),
        'high_similarity_pairs_ge_0_9':sum(abs(r['spearman_rho'])>=.9 for r in rows),
        'weak_primary_loadings_count':sum(r['weak_primary_loading'] for r in factor_rows),
        'cross_loading_count_at_0_3':sum(r['cross_loading_at_0_3'] for r in factor_rows),
        'factor_correlations': None if getattr(fa, 'phi_', None) is None else np.asarray(fa.phi_,dtype=float).tolist(),
    }
    eigen = pd.DataFrame({'factor_index':np.arange(1,len(names)+1),
                          'observed_eigenvalue':obs,'permuted_null95':null95,
                          'retained':obs>null95})
    return payload, pd.DataFrame(factor_rows), eigen, pd.DataFrame(rows,columns=['item_A','item_B','spearman_rho','threshold_0_8','threshold_0_9'])


def run(base):
    os.umask(0o077)
    work = base.expanduser().resolve()/PROJECT
    venv = work/'.venv_wsqs_n6039/bin/python'
    require(Path(sys.executable).resolve()==venv.resolve(), 'Run inside pinned N6039 virtualenv: '+str(venv))
    source = work/'00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    lockpath = work/'07_audit/COHORT_DATASET_LOCK_N6039.json'
    previous3 = work/'04_numerical_results/measurement/CORE8_AND_FEATURE_PREFLIGHT_N6039.json'
    previous5 = work/'07_audit/STAGE05_SENSITIVITY_SATISFACTION_N6039_COMPLETE.json'
    previous9 = work/'07_audit/STAGE09_SERVICE_RESPONSE_B200_N6039_COMPLETE.json'
    code=work/'02_code_private/reconciled/WSQS_Study_Repro_N6039'
    cfg=code/'config/feature_spec.json'
    result=work/'04_numerical_results/measurement'/STAGE
    audit=work/'07_audit'/(STAGE+'_COMPLETE.json')
    for path in (source, lockpath, previous3, previous5, previous9, cfg,
                 code/'src/wsqs/scoring.py', code/'src/wsqs/features.py',
                 code/'src/wsqs/extended.py'):
        require(path.is_file(), 'Missing prerequisite: '+str(path))
    require(sha(source)==SOURCE_SHA==load(lockpath)['output_sha256']['primary'], 'Frozen primary cohort SHA mismatch')
    require(load(previous3)['source_sha256']==SOURCE_SHA, 'Stage03 source mismatch')
    require(load(previous5)['status']=='PASS', 'Stage05 not complete')
    require(load(previous9)['status']=='PASS', 'Stage09 not complete')
    inputs={'source':SOURCE_SHA, 'code':sha(Path(__file__).resolve()),
            'feature_spec':sha(cfg), 'extended':sha(code/'src/wsqs/extended.py'),
            'scoring':sha(code/'src/wsqs/scoring.py'), 'features':sha(code/'src/wsqs/features.py'),
            'stage03':sha(previous3), 'stage05_audit':sha(previous5),'stage09_audit':sha(previous9),
            'B_ALPHA':B_ALPHA, 'B_PARALLEL':B_PARALLEL,
            'SEED_ALPHA':SEED_ALPHA, 'SEED_PARALLEL':SEED_PARALLEL}
    signature=hashlib.sha256(canonical(inputs).encode()).hexdigest()
    if audit.exists():
        prior=load(audit)
        require(prior['signature']==signature, 'Stage10 already exists with different algorithm signature')
        for fname, checksum in prior['output_sha256'].items():
            path=result/fname
            require(path.is_file() and sha(path)==checksum, 'Stage10 existing result changed: '+fname)
        print('PASS | Stage10 already complete; all output hashes verified, no rerun')
        return

    try:
        import factor_analyzer
    except ImportError as exc:
        raise RuntimeError("Missing factor_analyzer: in isolated venv run `python -m pip install 'factor_analyzer==0.5.1'` (do not alter base)") from exc
    sys.path.insert(0,str(code/'src'))
    from wsqs.features import construct_features
    from wsqs.extended import DOMAINS, alternative_outcomes
    from wsqs.models import partition_indices, model_settings

    print('='*77)
    print('STAGE10 | CORE8 MEASUREMENT, FACTORS, RELIABILITY, WEIGHTS, PROVIDER AUDIT')
    print('PRIMARY SHA256:',SOURCE_SHA)
    print('SIGNATURE:', signature)
    print('='*77,flush=True)
    df=pd.read_csv(source, dtype={'ResponseId':str},low_memory=False)
    require(df.shape==(N,182) and df.ResponseId.notna().all() and df.ResponseId.is_unique,
            'Unexpected primary cohort shape/unique IDs')
    if 'municipal_supply_applicable' not in df.columns:
        require('source09M_municipal_eligible' in df.columns, 'Missing original municipality evidence')
        df=df.rename(columns={'source09M_municipal_eligible':'municipal_supply_applicable'})
    spec=load(cfg)
    X, cat_cols, number_cols=construct_features(df)
    require(X.shape==(N,33) and len(cat_cols)+len(number_cols)==33, 'Feature architecture changed')
    fit,val,test,dev=partition_indices(N,int(model_settings()['primary_seed']))
    require((len(fit),len(val),len(test),len(dev))==(3864,967,1208,4831), 'Split protocol changed')
    severity=pd.DataFrame({d:pd.to_numeric(df[f'{d}_raw_severity'],errors='coerce') for d in DOMAINS})
    require(severity.stack().isin(range(1,6)).all() and severity.notna().sum(axis=1).between(7,8).all(),
            'CORE severities invalid')
    scores=(5.0-severity)*25
    equal=scores.mean(axis=1)
    require(np.isclose(equal.mean(),float(load(previous3)['core8_mean_full_precision']), atol=1e-11),
            'CORE8 mismatch Stage03')
    seven=severity.notna().sum(axis=1).eq(7)
    domain_rows=[]
    for d in DOMAINS:
        vals=scores[d].dropna()
        domain_rows.append({'domain':d,'n_valid':len(vals),'n_missing':N-len(vals),
                            'mean':float(vals.mean()), 'std':float(vals.std(ddof=1)),
                            'median':float(vals.median()),
                            'pct_at_0':float((vals==0).mean()),'pct_at_100':float((vals==100).mean())})
    core_summary={'N':N,'CORE8_mean':float(equal.mean()),'CORE8_sd':float(equal.std(ddof=1)),
                  'CORE8_median':float(equal.median()),
                  'CORE8_q025_q25_q75_q975':[float(z) for z in equal.quantile([.025,.25,.75,.975])],
                  'CORE8_min':float(equal.min()),'CORE8_max':float(equal.max()),
                  'eight_domain_n':int((~seven).sum()), 'seven_domain_n':int(seven.sum()),
                  'missing_by_domain':severity.isna().sum().astype(int).to_dict()}
    response_rows=[]
    for domain in DOMAINS:
        original=f'{domain}_raw_severity'
        for level in range(1,6):
            n=int((severity[domain]==level).sum())
            response_rows.append({'domain':domain,'item':original,'severity':level,
                                  'n':n,'pct_of_all':n/N,
                                  'pct_of_valid':n/int(severity[domain].notna().sum()),
                                  'domain_score':(5-level)*25})
    for domain in DOMAINS:
        response_rows.append({'domain':domain,'item':f'{domain}_raw_severity',
                              'severity':'UNAVAILABLE','n':int(severity[domain].isna().sum()),
                              'pct_of_all':float(severity[domain].isna().mean()),
                              'pct_of_valid':None,'domain_score':None})
    priority=pd.DataFrame({d:pd.to_numeric(df[f'{d}_priority_0_10'],errors='coerce') for d in DOMAINS})
    require(priority.notna().all().all() and priority.stack().between(0,10).all(), 'Invalid priorities')
    mean_full=priority.mean(axis=0)
    weights_full=mean_full/mean_full.sum()
    outcomes_dev, weights_dev, fallback=alternative_outcomes(df,dev)
    require(np.allclose(outcomes_dev['CORE8'], equal,rtol=0,atol=1e-12), 'CORE8 mismatch original scoring code')
    require(list(mean_full.index)==list(weights_dev.index), 'Weight label mismatch')
    weighted_full=severity.mul(weights_full).sum(axis=1) / severity.notna().mul(weights_full).sum(axis=1)
    weighted_full=(5-weighted_full)*25
    individual_den=priority.where(severity.notna()).sum(axis=1)
    individual=severity.mul(priority.where(severity.notna())).sum(axis=1) / individual_den.where(individual_den>0)
    individual=individual.where(individual_den>0, severity.mean(axis=1))
    individual=(5-individual)*25
    require(int((individual_den<=0).sum())==fallback, 'Individual zero-priority fallback mismatch')
    require(np.isfinite(weighted_full).all() and np.isfinite(individual).all(), 'Invalid weighting scores')
    priority_rows=[]
    for d in DOMAINS:
        p=priority[d]
        priority_rows.append({'domain':d,'mean_priority':float(mean_full[d]),'sd_priority':float(p.std(ddof=1)),
                              'median_priority':float(p.median()),
                              'weight_full_sample':float(weights_full[d]),
                              'weight_development_only':float(weights_dev[d]),'n':N})
    priority_freq=[]
    for d in DOMAINS:
        for value in range(11):
            n=int((priority[d]==value).sum())
            priority_freq.append({'domain':d,'priority_rating':value,'n':n,'pct':n/N})
    comparisons=[]
    for name,s in [('CORE8',equal), ('POP_WEIGHT_FULL_SAMPLE',weighted_full),
                   ('POP_WEIGHT_DEV_ONLY',outcomes_dev['POPULATION_PRIORITY']),
                   ('HOUSEHOLD_PRIORITY',individual),('CORE7A',outcomes_dev['CORE7A']),
                   ('CORE7B',outcomes_dev['CORE7B'])]:
        diff=s-equal
        comparisons.append({'variant':name, 'n':N, 'mean_score':float(s.mean()),
                            'sd_score':float(s.std(ddof=1)),
                            'mean_absolute_change_vs_CORE8':float(diff.abs().mean()),
                            'spearman_vs_CORE8':float(spearmanr(equal,s).statistic),
                            'max_absolute_change':float(diff.abs().max())})
    print('PASS | CORE-8, 8 domains, priority weights and score sensitivities')
    cc=severity.dropna()
    require(len(cc)>=5000, 'Unexpected CORE8 complete-case cohort size')
    x=cc.to_numpy(dtype=float)
    imputed=severity.fillna(severity.median()).to_numpy(dtype=float)
    standard,rbar=stan_alpha(x)
    spearman_mean=float(cc.corr(method='spearman').to_numpy()[np.triu_indices(8,1)].mean())
    reliability={'complete_case_n':len(cc),'alpha_raw_complete_case':alpha(x),
                 'alpha_raw_complete_case_CI95':bootstrap_alpha(x,B_ALPHA,SEED_ALPHA),
                 'alpha_raw_median_imputed':alpha(imputed),
                 'alpha_raw_median_imputed_CI95':bootstrap_alpha(imputed,B_ALPHA,SEED_ALPHA+1),
                 'alpha_standardized_complete_case':standard,
                 'mean_pairwise_pearson_r':rbar,
                 'mean_pairwise_spearman_rho':spearman_mean,
                 'bootstrap_resamples':B_ALPHA, 'bootstrap_seed':SEED_ALPHA,
                 'interpretation':'Descriptive internal coherence, not external criterion validation'}
    print('PASS | CORE8 reliability and B1000 alpha bootstrap')
    # Provider coding must preserve distinctions between evidence and true dependence.
    evidence=df.municipal_supply_applicable.astype(str).str.strip().str.lower().isin(['true','yes','1'])
    q11='Q 1.1_raw_value'
    require(q11 in df.columns, 'Missing original domestic source Q1.1')
    provision=df[q11].astype('string').fillna('<MISSING>').str.strip()
    municipality_primary=provision.str.contains('municipal|corporation',case=False,regex=True).to_numpy(bool)
    if municipality_primary.sum()==0:
        # Do not assume source labels without inspecting the source distribution.
        raise RuntimeError('Cannot recognize municipality/corporation category in Q1.1')
    require(not (municipality_primary & (~evidence.to_numpy())).any(),
            'Domestic municipal primary but provider evidence false; check classification')
    provider_summary={'N':N,'municipal_positive_evidence':int(evidence.sum()),
                      'outside_positive_evidence':int((~evidence).sum()),
                      'municipal_primary_domestic':int(municipality_primary.sum()),
                      'municipal_evidence_nonprimary_domestic':int((evidence.to_numpy() & ~municipality_primary).sum()),
                      'source_note':'Absence of positive municipal evidence is NOT proof of nonuse or absence of any provider'}
    provider_rows=[]
    for group, mask in [('positive_evidence',evidence),('no_positive_evidence',~evidence),
                        ('primary_municipal',pd.Series(municipality_primary,index=df.index)),
                        ('other_primary_with_municipal_evidence',evidence & ~pd.Series(municipality_primary,index=df.index))]:
        sub=df.loc[mask]
        for question in ('Q 9.1_raw_value','Q 9.2_raw_value','Q 9.3_raw_value','Q 9.4_raw_value'):
            if question in df.columns:
                provider_rows.extend(category_counts(sub, question.replace('_raw_value',''),question,scope=group))
        for q in ['service_Q3_5','service_Q3_6','service_Q4_2','service_Q4_3','service_Q8_2','service_Q9_2','service_Q9_4']:
            if q in X:
                xsub=X.loc[mask,q]
                counts=xsub.astype('string').fillna('<MISSING>').value_counts(dropna=False)
                for status,c in counts.items():
                    provider_rows.append({'scope':group,'source':q,'source_column':q,
                                          'response':str(status),'n':int(c),'fraction':float(c/len(sub))})
    provider_joint=[]
    for primary in sorted(provision.unique()):
        for ev in [False,True]:
            idx=provision.eq(primary) & evidence.eq(ev)
            provider_joint.append({'primary_domestic_Q1_1':str(primary),
                                   'positive_municipal_use_evidence_Q1_1_to_Q1_3':bool(ev),
                                   'n':int(idx.sum()),'pct_full':float(idx.mean())})
    q91=pd.to_numeric(df['Customer_service_raw_severity'],errors='coerce')
    q92=pd.to_numeric(df['Q9_2_complaint_resolution_raw_severity'], errors='coerce')
    provider_status=[]
    for flag in [False, True]:
        mask=evidence.eq(flag)
        provider_status.append({'has_positive_municipal_evidence':bool(flag),
                                'n_households':int(mask.sum()),
                                'q9_1_customer_service_valid_rating_n':int(q91[mask].notna().sum()),
                                'q9_1_customer_service_unavailable_n':int(q91[mask].isna().sum()),
                                'q9_2_complaint_resolution_valid_rating_n':int(q92[mask].notna().sum()),
                                'q9_2_complaint_resolution_unavailable_n':int(q92[mask].isna().sum()),
                                'q9_2_absence_not_treated_as_never_complained':True})
    provider_summary['q9_status_by_evidence']=provider_status
    q91_raw_col='Q 9.1_raw_value'
    if q91_raw_col in df.columns:
        text=df[q91_raw_col].astype('string').str.lower().str.strip().fillna('')
        explicit_inhouse=text.str.contains('full water system in-house',regex=False)
        require((~q91.notna() | ~explicit_inhouse).all(),
                'Q9.1 full-in-house response assigned numeric severity')
        provider_summary['Q9_1_explicit_full_inhouse_N']=int(explicit_inhouse.sum())
        provider_summary['Q9_1_genuinely_unanswered_or_other_unscored_N']=int(q91.isna().sum()-explicit_inhouse.sum())
        require(provider_summary['Q9_1_genuinely_unanswered_or_other_unscored_N']>=0,
                'Q9.1 full-inhouse exceeds missing severity')
    else:
        provider_summary['Q9_1_explicit_full_inhouse_N']='RAW_COLUMN_NOT_PRESENT_CHECK_UPSTREAM'
    # All 22 service predictors' recorded survey response frequencies for PI.
    all_support=[]
    for domain, mapping in spec['domains'].items():
        for q,col in mapping.items():
            require(col in df, 'Missing service support input '+col)
            all_support.extend(category_counts(df,f'{domain} {q}',col))
    all_provision=category_counts(df,'Q1.1 primary domestic provision',q11)
    all_support.extend(all_provision)
    print('PASS | Provider evidence, Q9.1/Q9.2 distinctions and 22 support response frequencies')
    profile_rows=[]
    for group in sorted(provision.unique()):
        mask=provision.eq(group)
        d={'primary_domestic_provision':str(group),'n':int(mask.sum()),
           'CORE8_mean':float(equal[mask].mean()), 'CORE8_std':float(equal[mask].std(ddof=1)),
           'WSQS_le_70_n':int((equal[mask]<=70).sum()),
           'WSQS_le_70_fraction':float((equal[mask]<=70).mean())}
        for domain in DOMAINS:
            d[f'{domain}_score_mean']=float(scores.loc[mask,domain].mean())
            d[f'{domain}_score_valid_n']=int(scores.loc[mask,domain].notna().sum())
        profile_rows.append(d)
    # Private, exact row-level source for ECDFs and downstream figure rebuilding.
    private_scores=pd.DataFrame({'ResponseId':df.ResponseId, 'CORE8':equal,
        'POP_WEIGHT_FULL_SAMPLE':weighted_full,
        'POP_WEIGHT_DEV_ONLY':outcomes_dev['POPULATION_PRIORITY'],
        'HOUSEHOLD_PRIORITY':individual,
        'CORE7A':outcomes_dev['CORE7A'], 'CORE7B':outcomes_dev['CORE7B']})

    # Source of factor comparison: 8 CORE +21 ordinal supports+contact Q9.3.
    factor_summary,loadings,eigen,pairs=make_factor(df,spec,severity,result)
    print('PASS | Factor diagnostics: 30 items, minres oblimin, KMO/Bartlett/parallel B300; retained:',factor_summary['factors_retained'])
    results={
        '01_core8_measurement_summary.json':core_summary,
        '02_domain_level_summary.csv':pd.DataFrame(domain_rows),
        '03_core8_question_response_frequencies.csv':pd.DataFrame(response_rows),
        '04_priority_domains_and_weights.csv':pd.DataFrame(priority_rows),
        '05_stated_priority_response_frequencies.csv':pd.DataFrame(priority_freq),
        '06_weighting_and_core7_agreement.csv':pd.DataFrame(comparisons),
        '07_core8_alpha_and_bootstrap.json':reliability,
        '08_factor_structure_summary.json':factor_summary,
        '09_factor_loadings_30_items.csv':loadings,
        '10_factor_parallel_analysis_eigenvalues.csv':eigen,
        '11_factor_spearman_high_similarity_pairs.csv':pairs,
        '12_provider_evidence_summary.json':provider_summary,
        '13_provider_Q9_and_fitted_item_response_status.csv':pd.DataFrame(provider_rows),
        '14_provider_by_primary_domestic_source.csv':pd.DataFrame(provider_joint),
        '15_service_support_22_original_response_frequencies.csv':pd.DataFrame(all_support),
        '16_weighting_household_scores_PRIVATE.csv':private_scores,
        '17_primary_provision_domain_profiles.csv':pd.DataFrame(profile_rows),
    }
    require(not result.exists() or not any(result.iterdir()),'Nonempty Stage10 output directory without completed audit; manual inspection required')
    result.mkdir(parents=True, exist_ok=True)
    for fname,data in results.items():
        target=result/fname
        if isinstance(data, pd.DataFrame):
            write_csv(target,data)
        else:
            write_json(target,data)
    sums={fname:sha(result/fname) for fname in results}
    final={'status':'PASS','stage':STAGE,'signature':signature,'frozen_source_sha256':SOURCE_SHA,
           'input_sha256':inputs,'n_households':N,'n_models_fitted':0,
           'output_sha256':sums,'analytical_inventory':'30 ordinal diagnostics; Q2.8 and Q6.5 excluded; Q9.3 retained',
           'selection_and_uncertainty':'minres oblimin, parallel 300 column permutations; alpha 1000 bootstrap',
           'provider_applicability_policy':'audit only; Q9.1 structural NA kept distinct; Q9.2 not recoded as no complaint',
           'results_dir':str(result)}
    write_json(audit,final)
    print('='*77)
    print('PASS | Stage10 complete; 17 full-precision source files; no models fitted')
    print('CORE8 mean:',format(equal.mean(),'.9f'),'7-domain n:',int(seven.sum()))
    print('CORE8 alpha:',format(reliability['alpha_raw_complete_case'],'.6f'), 'KMO:',format(factor_summary['overall_kmo'],'.6f'))
    print('Provider positive municipal evidence n:',provider_summary['municipal_positive_evidence'])
    print('RESULTS:',result)
    print('AUDIT:',audit)
    print('='*77)


if __name__=='__main__':
    try:
        workbase=Path.home()/'Desktop/Spring 2026/Chapter 3 MANUAL RERUN'
        if len(sys.argv)>1:
            require(len(sys.argv)==3 and sys.argv[1]=='--base','Usage: script.py [--base <original Chapter 3 root>]')
            workbase=Path(sys.argv[2])
        run(workbase)
    except Exception as exc:
        print('FAILED |',type(exc).__name__,str(exc),file=sys.stderr)
        sys.exit(1)
