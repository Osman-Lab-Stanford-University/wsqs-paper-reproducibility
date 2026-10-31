#!/usr/bin/env python3
"""WSQS Stage 09: frozen service-response profiles and B200 refit uncertainty.

Run using the isolated scientific environment:

  WORK="$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_WSQS_RERUN_20261030_N6039"
  "$WORK/.venv_wsqs_n6039/bin/python" ~/Downloads/WSQS_STAGE09_SERVICE_RESPONSE_B200_N6039.py

Scientific specification: four pre-established service-condition/priority pairs;
severity 1--5 assigned to one predictor at a time; all remaining predictor
values kept as observed on the exact frozen held-out sample. Q4.2 pressure is
restricted to test households with substantive municipal-service pressure
values. Quartiles of paired priorities are computed from original N6039 cohort.

200 resamples WITH replacement of the full development set (n4831), one
fixed-iteration CatBoost refit per resample and evaluation against exactly the
same fixed held-out scenario pool. Iteration count is taken from Stage04's
validated original full model (500 for N6039), NOT the old N6145 value (253).
B200 refitting uncertainty is distinct from Stage08's conditional SHAP B1000
and Stage07's priority-null B1000.

Two additional pairs (water appearance/clarity) are evaluated only using the
frozen model, with NO refit intervals, matching the prior scientific scope.

Never modifies historical data or models. Checkpoints are write-once and
cryptographically verified before reuse. Record-level data and bootstrap
membership indices are PRIVATE; do not release them publicly.

Note: the October 19 public package excludes the original private B200 RNG
loop. This implementation preserves the documented B200 bootstrap design and
uses an explicit, recorded SeedSequence stream; it does NOT claim byte-level
identity to the old private RNG draws.
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
STAGE = 'STAGE09_SERVICE_RESPONSE_B200_N6039'
SOURCE_SHA = 'c72b8480d3cff223d1d4cf0ebcc29a3dd80204d563cd63a4295f6739f0ed77a1'
N = 6039
DEV_N = 4831
TEST_N = 1208
B = 200
BOOT_SEED = 20260714
PAIRS = (
    ('Complaint resolution', 'service_Q9_2', 'Customer_service_priority_0_10', 'numeric'),
    ('Value for money', 'service_Q2_5', 'Affordability_priority_0_10', 'numeric'),
    ('Water pressure', 'service_Q4_2', 'Reliability_priority_0_10', 'categorical'),
    ('Shortage frequency', 'service_Q8_3', 'Quantity_priority_0_10', 'numeric'),
)
FROZEN_ONLY = (
    ('Water appearance', 'service_Q6_2', 'Acceptability_priority_0_10'),
    ('Water clarity', 'service_Q7_3', 'Quality_priority_0_10'),
)


def check(condition, error):
    if not condition:
        raise RuntimeError(error)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write_json_new(path, obj):
    path = Path(path)
    check(not path.exists(), 'Will not overwrite: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.partial', encoding='utf-8', delete=False) as f:
        tmp = Path(f.name)
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)


def write_csv_new(path, data):
    path = Path(path)
    check(not path.exists(), 'Will not overwrite: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.partial', encoding='utf-8', delete=False) as f:
        tmp = Path(f.name)
        data.to_csv(f, index=False, float_format='%.17g')
    os.replace(tmp, path)


def category_value(feature, level):
    check(level in (1, 2, 3, 4, 5), 'Not a substantive severity category')
    return f'severity_{level}' if feature == 'service_Q4_2' else float(level)


def quantile_settings(X):
    out = {}
    for label, feature, priority, mode in PAIRS:
        vals = pd.to_numeric(X[priority], errors='coerce')
        check(vals.notna().all() and vals.between(0, 10).all(), 'Invalid priorities ' + priority)
        lo, hi = np.quantile(vals.to_numpy(dtype=float), [0.25, 0.75])
        check(lo <= hi, 'Invalid priority quartiles for ' + label)
        out[label] = {'feature': feature, 'priority': priority, 'mode': mode,
                      'low': float(lo), 'high': float(hi)}
    return out


def evaluation_masks(X, test):
    on_test = X.iloc[test]
    pressure = on_test['service_Q4_2'].astype(str).str.fullmatch(r'severity_[1-5]').to_numpy(dtype=bool)
    check(0 < int(pressure.sum()) <= len(test), 'Pressure eligibility collapsed')
    check(np.array_equal(pressure, on_test['service_Q4_2'].isin(
        [f'severity_{k}' for k in range(1,6)]).to_numpy()), 'Pressure eligibility disagreement')
    return {'Water pressure': pressure}


def make_scenario_pool(X_test, categories, qsettings, masks):
    """Build a single fixed Pool for all B200 refit evaluations.

    Each scenario changes ONLY its nominated feature, optionally also its
    associated priority. The same ordered heldout rows appear in each block.
    """
    frames, slices = [], {}
    position = 0
    for label, feature, priority, mode in PAIRS:
        cohort_mask = masks.get(label, np.ones(len(X_test), dtype=bool))
        base = X_test.iloc[np.flatnonzero(cohort_mask)].copy()
        check(len(base) > 0, 'Empty test population for ' + label)
        for level in (1,2,3,4,5):
            scenario = base.copy()
            scenario[feature] = category_value(feature, level)
            key = (label, 'profile', level)
            frames.append(scenario)
            slices[key] = (position, position + len(scenario))
            position += len(scenario)
        for tag in ('low', 'high'):
            for level in (1,5):
                scenario = base.copy()
                scenario[feature] = category_value(feature, level)
                scenario[priority] = qsettings[label][tag]
                key = (label, tag, level)
                frames.append(scenario)
                slices[key] = (position, position + len(scenario))
                position += len(scenario)
    combined = pd.concat(frames, axis=0, ignore_index=True)
    check(len(combined) == position and list(combined.columns) == list(X_test.columns),
          'Scenario concatenation changed the 33 features')
    return Pool(combined, cat_features=categories), slices


def collect_means(predictions, slices):
    predicted = np.asarray(predictions, dtype=float)
    check(np.isfinite(predicted).all() and len(predicted) == max(end for _,end in slices.values()),
          'Scenario predictions incomplete/nonfinite')
    out = {}
    for key, (start,end) in slices.items():
        out['|'.join(str(part) for part in key)] = float(predicted[start:end].mean())
    return out


def predict_scenarios(model, pool, slices):
    return collect_means(model.predict(pool), slices)


def key_for(label, kind, level):
    return f'{label}|{kind}|{level}'


def refit_draw(dev, i):
    check(1 <= i <= B, 'Replicate index outside B200')
    generator = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, i]))
    return np.asarray(dev, dtype=np.int64)[generator.integers(0, len(dev), size=len(dev))]


def refit_params(settings, tree_count):
    params = dict(settings['catboost'])
    params['iterations'] = int(tree_count)
    params.pop('od_type', None)
    params.pop('od_wait', None)
    params.update(verbose=False, allow_writing_files=False)
    return params


def fit_one(i, development, prepared, y, cats, params, pool, slices, checkpoint_root, signature):
    target_dir = checkpoint_root / f'refit_{i:04d}'
    if target_dir.exists():
        manifest_file = target_dir / 'refit_manifest.json'
        check(manifest_file.is_file(), 'Incomplete refit checkpoint: ' + str(target_dir))
        meta = read_json(manifest_file)
        check(meta['signature'] == signature and meta['replicate'] == i,
              'Wrong refit signature/index at ' + str(target_dir))
        for name, expected in meta['output_sha256'].items():
            check((target_dir/name).is_file() and sha256(target_dir/name) == expected,
                  'Corrupt refit: ' + str(target_dir/name))
        result = read_json(target_dir/'scenario_means.json')
        check(result['replicate'] == i and result['signature'] == signature,
              'Refit results out of sync at ' + str(target_dir))
        return result['predicted_means'], False

    draw = refit_draw(development, i)
    check(len(draw) == DEV_N and np.isin(draw, development).all(), 'Invalid development bootstrap draw')
    fitted = CatBoostRegressor(**params)
    fitted.fit(Pool(prepared.iloc[draw], y.iloc[draw], cat_features=cats))
    means = predict_scenarios(fitted, pool, slices)

    tmp = Path(tempfile.mkdtemp(prefix=f'refit_{i:04d}.partial.', dir=checkpoint_root))
    try:
        fitted.save_model(str(tmp / 'refit_model.cbm'))
        np.save(tmp / 'development_sample_indices_PRIVATE.npy', draw, allow_pickle=False)
        (tmp/'scenario_means.json').write_text(
            json.dumps({'signature': signature, 'replicate': i, 'predicted_means': means},
                       indent=2, allow_nan=False)+'\n', encoding='utf-8')
        names = ['refit_model.cbm', 'development_sample_indices_PRIVATE.npy', 'scenario_means.json']
        manifest = {'status':'PASS', 'signature':signature, 'replicate':i,
                    'development_n': len(draw), 'tree_count': params['iterations'],
                    'refit_method': 'full development bootstrap with replacement, fixed validated tree count',
                    'rng':f'SeedSequence([{BOOT_SEED},{i}]); numpy.default_rng',
                    'output_sha256': {name:sha256(tmp/name) for name in names}}
        (tmp/'refit_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n',encoding='utf-8')
        os.replace(tmp, target_dir)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return means, True


def percentile_ci(v):
    vals = np.asarray(v, dtype=float)
    check(len(vals)==B and np.isfinite(vals).all(), 'B200 refit vector missing values')
    result=np.quantile(vals,[.025,.975])
    return float(result[0]), float(result[1])


def summarize(frozen_means, replicate_means, qsettings, masks, test_n, data_sha, signature):
    check(len(replicate_means)==B, 'Incorrect B200 replicate count')
    records = pd.DataFrame(replicate_means)
    check(records.shape[0] == B and records.notna().all().all(), 'B200 refit results incomplete')
    profile, adjacent, main, modification, eligibility = [],[],[],[],[]
    for label, feature, priority, mode in PAIRS:
        ntest = int(masks[label].sum()) if label in masks else test_n
        levels = [key_for(label,'profile',k) for k in range(1,6)]
        check(all(k in records.columns and k in frozen_means for k in levels),
              'Missing severity profile for '+label)
        point=np.array([frozen_means[k] for k in levels])
        refit=records[levels].to_numpy(dtype=float)
        for level in range(1,6):
            lo, hi=percentile_ci(refit[:,level-1])
            profile.append({'service_condition':label, 'service_feature':feature,
                            'severity':level,'evaluation_n':ntest,
                            'frozen_model_predicted_mean_wsqs_100':point[level-1],
                            'bootstrap_ci2p5':lo,'bootstrap_ci97p5':hi,
                            'bootstrap_refits':B})
        changes=np.diff(point)
        refit_changes=np.diff(refit,axis=1)
        largest_index=int(np.argmax(np.abs(changes)))
        top=np.argmax(np.abs(refit_changes),axis=1)
        for j in range(4):
            lo,hi=percentile_ci(refit_changes[:,j])
            adjacent.append({'service_condition':label,'transition':f'{j+1}-{j+2}',
                             'severity_from':j+1,'severity_to':j+2,
                             'evaluation_n':ntest,
                             'signed_point_change_wsqs_100':float(changes[j]),
                             'signed_ci2p5':lo,'signed_ci97p5':hi,
                             'absolute_point_change_wsqs_100':float(abs(changes[j])),
                             'largest_abs_change_refit_count':int(np.sum(top==j)),
                             'largest_abs_change_refit_pct':100*float(np.mean(top==j)),
                             'bootstrap_refits':B})
        check(sum(a['largest_abs_change_refit_count'] for a in adjacent if a['service_condition']==label)==B,
              'Adjacent largest-change counts do not total B200 for '+label)
        end=refit[:,4]-refit[:,0]
        mid=refit[:,2]-refit[:,0]
        e_lo,e_hi=percentile_ci(end)
        m_lo,m_hi=percentile_ci(mid)
        largest_abs_draw=np.abs(refit_changes[:,largest_index])
        l_lo,l_hi=percentile_ci(largest_abs_draw)
        lows=np.array([records[key_for(label,'low',k)].to_numpy(dtype=float) for k in (1,5)]).T
        highs=np.array([records[key_for(label,'high',k)].to_numpy(dtype=float) for k in (1,5)]).T
        low_contrast=lows[:,1]-lows[:,0]
        high_contrast=highs[:,1]-highs[:,0]
        mods=high_contrast-low_contrast
        mod_lo,mod_hi=percentile_ci(mods)
        freeze_lo=(frozen_means[key_for(label,'low',5)]-frozen_means[key_for(label,'low',1)])
        freeze_hi=(frozen_means[key_for(label,'high',5)]-frozen_means[key_for(label,'high',1)])
        low_ci=percentile_ci(low_contrast)
        high_ci=percentile_ci(high_contrast)
        mod=freeze_hi-freeze_lo
        main.append({'service_condition':label,'evaluation_n':ntest,
                     'severity5_minus_1_point_wsqs_100':float(point[4]-point[0]),
                     'severity5_minus_1_ci2p5':e_lo,'severity5_minus_1_ci97p5':e_hi,
                     'severity3_minus_1_point_wsqs_100':float(point[2]-point[0]),
                     'severity3_minus_1_ci2p5':m_lo,'severity3_minus_1_ci97p5':m_hi,
                     'largest_abs_adjacent_transition':f'{largest_index+1}-{largest_index+2}',
                     'largest_abs_adjacent_point_wsqs_100':float(abs(changes[largest_index])),
                     'largest_abs_adjacent_ci2p5':l_lo,'largest_abs_adjacent_ci97p5':l_hi,
                     'priority_25th':qsettings[label]['low'],
                     'priority_75th':qsettings[label]['high'],
                     'priority_modification_wsqs_100':mod,
                     'priority_modification_ci2p5':mod_lo,'priority_modification_ci97p5':mod_hi,
                     'bootstrap_refits':B})
        modification.append({'service_condition':label,'paired_priority':priority,
                             'priority_25th':qsettings[label]['low'],
                             'priority_75th':qsettings[label]['high'],
                             'evaluation_n':ntest,
                             'severity5_minus1_at_low_priority':freeze_lo,
                             'low_priority_ci2p5':low_ci[0],
                             'low_priority_ci97p5':low_ci[1],
                             'severity5_minus1_at_high_priority':freeze_hi,
                             'high_priority_ci2p5':high_ci[0],
                             'high_priority_ci97p5':high_ci[1],
                             'modification_high_minus_low':mod,
                             'modification_ci2p5':mod_lo,
                             'modification_ci97p5':mod_hi,
                             'bootstrap_refits':B})
        eligibility.append({'service_condition':label,'feature':feature,
                            'eligible_heldout_households':ntest,
                            'heldout_total':test_n,
                            'eligibility_rule': 'substantive Q4.2 severity_1 to severity_5; municipal use applicable' if label=='Water pressure' else 'all frozen heldout households',
                            'priority_25th':qsettings[label]['low'],
                            'priority_75th':qsettings[label]['high']})
    return (pd.DataFrame(profile), pd.DataFrame(main), pd.DataFrame(adjacent),
            pd.DataFrame(modification),pd.DataFrame(eligibility), records)


def frozen_only_pairs(model, X_test, cats, X_full):
    rows=[]
    for label, feature, priority in FROZEN_ONLY:
        lo,hi=np.quantile(pd.to_numeric(X_full[priority],errors='raise').to_numpy(dtype=float),[.25,.75])
        means={}
        for tag,pval in (('low',lo),('high',hi)):
            for level in (1,5):
                s=X_test.copy()
                s[feature]=float(level)
                s[priority]=float(pval)
                means[tag,level]=float(np.mean(model.predict(Pool(s,cat_features=cats))))
        rows.append({'service_condition':label,'feature':feature,'priority':priority,
                     'evaluation_n':len(X_test),'priority_25th':float(lo),'priority_75th':float(hi),
                     'severity5_minus1_low_priority':means['low',5]-means['low',1],
                     'severity5_minus1_high_priority':means['high',5]-means['high',1],
                     'modification_high_minus_low':(means['high',5]-means['high',1])-(means['low',5]-means['low',1]),
                     'uncertainty': 'not calculated; frozen model only',
                     'bootstrap_refits':0})
    return pd.DataFrame(rows)


def smoke_test():
    rng=np.random.default_rng(4)
    n=110
    X=pd.DataFrame({'service_Q9_2':rng.integers(1,6,n).astype(float),
                    'service_Q2_5':rng.integers(1,6,n).astype(float),
                    'service_Q4_2':rng.choice([f'severity_{k}' for k in range(1,6)], n),
                    'service_Q8_3':rng.integers(1,6,n).astype(float),
                    'context_primary_provision':rng.choice(['M','G'],n),
                    'Affordability_priority_0_10':rng.integers(0,11,n).astype(float),
                    'Reliability_priority_0_10':rng.integers(0,11,n).astype(float),
                    'Quantity_priority_0_10':rng.integers(0,11,n).astype(float),
                    'Customer_service_priority_0_10':rng.integers(0,11,n).astype(float)})
    cats=['service_Q4_2','context_primary_provision']
    y=pd.Series(75-2*X.service_Q9_2-2*X.service_Q2_5+rng.normal(0,4,n))
    m=CatBoostRegressor(iterations=8, depth=3, random_seed=4, verbose=False, allow_writing_files=False)
    m.fit(Pool(X,y,cat_features=cats))
    Xtest=X.iloc[:45].copy()
    qs=quantile_settings(X)
    masks=evaluation_masks(X,np.arange(45))
    pool,slices=make_scenario_pool(Xtest,cats,qs,masks)
    result=predict_scenarios(m,pool,slices)
    check(len(result)==36,'Expected four pairs x nine scenarios')
    check(all(np.isfinite(list(result.values()))),'Nonfinite test predictions')
    replica=[]
    for i in (1,2):
        gen=np.random.default_rng(np.random.SeedSequence([BOOT_SEED,i]))
        idx=gen.integers(0,len(X),len(X))
        mm=CatBoostRegressor(iterations=8, depth=3, random_seed=4, verbose=False,allow_writing_files=False)
        mm.fit(Pool(X.iloc[idx],y.iloc[idx],cat_features=cats))
        replica.append(predict_scenarios(mm,pool,slices))
    check(len(replica)==2 and list(replica[0])==list(result), 'Replicate key mismatch')
    print('PASS | Stage09 synthetic profile, categorical pressure, quartiles and B2 refit smoke test')
    print('NO empirical household data accessed.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path.home()/'Desktop/Spring 2026/Chapter 3 MANUAL RERUN')
    parser.add_argument('--self-test', action='store_true')
    args=parser.parse_args()
    if args.self_test:
        smoke_test()
        return
    os.umask(0o077)
    work=args.base.expanduser().resolve()/PROJECT
    expected_python=(work/'.venv_wsqs_n6039/bin/python').resolve()
    check(Path(sys.executable).resolve()==expected_python,
          'Run with isolated WSQS environment: '+str(expected_python))
    data_path=work/'00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    lock_path=work/'07_audit/COHORT_DATASET_LOCK_N6039.json'
    stage04_audit_path=work/'07_audit/STAGE04_PRIMARY_CROSSDOMAIN_N6039_COMPLETE.json'
    stage08_audit_path=work/'07_audit/STAGE08_SHAP_ATTRIBUTION_N6039_COMPLETE.json'
    original_model_dir=work/'03_run/checkpoints/STAGE04_PRIMARY_CROSSDOMAIN_N6039/BLOCK_SCP'
    model_path=original_model_dir/'final_model.cbm'
    manifest_path=original_model_dir/'model_manifest.json'
    predictions_path=original_model_dir/'heldout_predictions.csv'
    code=work/'02_code_private/reconciled/WSQS_Study_Repro_N6039'
    settings_path=code/'config/models.json'
    plans_path=code/'config/scientific_plans.json'
    all_required=[data_path,lock_path,stage04_audit_path,stage08_audit_path,
                  model_path,manifest_path,predictions_path,settings_path,plans_path,
                  code/'src/wsqs/features.py',code/'src/wsqs/models.py',code/'config/feature_spec.json']
    for p in all_required:
        check(p.is_file(),'Missing prerequisite '+str(p))
    check(sha256(data_path)==SOURCE_SHA==read_json(lock_path)['output_sha256']['primary'],
          'Frozen N6039 dataset SHA changed')
    stage04_audit=read_json(stage04_audit_path)
    stage08_audit=read_json(stage08_audit_path)
    check(stage04_audit['status']=='PASS' and stage08_audit['status']=='PASS',
          'Stage04/08 audit must have passed')
    meta=read_json(manifest_path)
    check(meta['signature']==stage04_audit['signature'], 'Frozen Stage04 signature changed')
    for name in ('final_model.cbm','heldout_predictions.csv','validation_history.json'):
        check(sha256(original_model_dir/name)==meta['output_sha256'][name],
              'Stage04 model checkpoint altered: '+name)
    sources=[settings_path,plans_path,code/'config/feature_spec.json',
             code/'src/wsqs/features.py',code/'src/wsqs/models.py',code/'src/wsqs/extended.py']
    signature_inputs={
        'source_sha256':SOURCE_SHA,
        'original_stage04_signature':stage04_audit['signature'],
        'stage08_signature':stage08_audit['signature'],
        'model_sha256':sha256(model_path),
        'model_manifest_sha256':sha256(manifest_path),
        'script_sha256':sha256(Path(__file__).resolve()),
        'scientific_code_sha256':{str(p.relative_to(work)):sha256(p) for p in sources},
        'B':B,'bootstrap_seed':BOOT_SEED,
        'sample_rng':'numpy.random.default_rng(np.random.SeedSequence([20260714, replicate]))',
        'refit_sample':'n4831 development observations resampled with replacement',
        'model_iterations':'Stage04 validated fixed tree count',
    }
    signature=hashlib.sha256(canonical_json(signature_inputs).encode()).hexdigest()
    # Keep an immutable copy of the exact executable inside the new private project.
    runner_archive=work/'02_code_private/stage_runners'/Path(__file__).name
    runner_archive.parent.mkdir(parents=True, exist_ok=True)
    if runner_archive.exists():
        check(sha256(runner_archive)==sha256(Path(__file__).resolve()),
              'A different Stage09 script is already frozen in the project')
    elif Path(__file__).resolve()!=runner_archive.resolve():
        shutil.copy2(Path(__file__).resolve(), runner_archive)
        check(sha256(runner_archive)==sha256(Path(__file__).resolve()),
              'Project script archive copy failed')
    results=work/'04_numerical_results/inference'/STAGE
    checkpoints=work/'03_run/checkpoints'/STAGE
    audit_path=work/'07_audit'/(STAGE+'_COMPLETE.json')
    output_names=('01_S28_response_profiles_with_B200_intervals.csv',
                  '02_MAIN_TABLE07_priority_response_contrasts.csv',
                  '03_S29_adjacent_severity_changes_with_B200_frequency.csv',
                  '04_S7_priority_modification_low_high_B200.csv',
                  '05_two_frozen_only_pairs_no_B200.csv',
                  '06_service_response_eligibility_and_quartiles.csv',
                  '07_B200_all_refit_scenario_means_PRIVATE.csv',
                  '08_service_response_analysis_manifest.json')
    if audit_path.exists():
        audit=read_json(audit_path)
        check(audit['status']=='PASS' and audit['signature']==signature,
              'Stage09 completion audit does not match this exact code/model/cohort')
        for filename,expected in audit['result_sha256'].items():
            check((results/filename).is_file() and sha256(results/filename)==expected,
                  'Stage09 result changed: '+filename)
        for i in range(1,B+1):
            p=checkpoints/f'refit_{i:04d}'/'refit_manifest.json'
            check(p.is_file() and read_json(p)['signature']==signature, 'Missing refit '+str(i))
        print('PASS | Stage09 complete; 200 refits and full-precision outputs verified; no model refit')
        print('RESULTS:',results)
        return
    for name in output_names:
        check(not (results/name).exists(), 'Stage09 partially generated aggregates without audit: '+name)
    results.mkdir(parents=True,exist_ok=True)
    checkpoints.mkdir(parents=True,exist_ok=True)

    sys.path.insert(0,str(code/'src'))
    from wsqs.features import construct_features
    from wsqs.models import model_settings, partition_indices, prepare_catboost
    from wsqs.extended import scientific_plans,DOMAINS
    settings=model_settings()
    plans=scientific_plans()
    df=pd.read_csv(data_path,low_memory=False,dtype={'ResponseId':str})
    check(df.shape==(N,182) and df.ResponseId.notna().all() and df.ResponseId.is_unique,
          'Frozen dataset dimensions or ResponseIds invalid')
    if 'municipal_supply_applicable' not in df.columns:
        check('source09M_municipal_eligible' in df.columns,'No provider eligibility field')
        df=df.rename(columns={'source09M_municipal_eligible':'municipal_supply_applicable'})
    X,cats,nums=construct_features(df)
    check(X.shape==(N,33) and list(X.columns)==plans['full']==meta['features'],
          'Changed predictor specification or order')
    check(set(cats)==set(meta['categorical']) and set(nums)==set(meta['numeric']),
          'Changed categorical/numeric predictor typing')
    raw=df[[f'{d}_raw_severity' for d in DOMAINS]].apply(pd.to_numeric,errors='coerce')
    check(raw.notna().sum(axis=1).between(7,8).all(), 'Changed domain availability')
    y=pd.Series((5-raw.mean(axis=1))*25,dtype=float)
    check(y.notna().all() and y.between(0,100).all(), 'Invalid CORE8 outcome')
    fit,val,test,dev=partition_indices(N,20260714)
    check((len(fit),len(val),len(test),len(dev))==(3864,967,TEST_N,DEV_N),
          'Stage04 partition definition changed')
    old_predictions=pd.read_csv(predictions_path,dtype={'ResponseId':str})
    check(np.array_equal(old_predictions.canonical_row_index.to_numpy(dtype=int),test),
          'Frozen heldout order differs from Stage04')
    check(old_predictions.ResponseId.tolist()==df.ResponseId.iloc[test].astype(str).tolist(),
          'Frozen heldout ResponseIds differ')
    check(np.allclose(old_predictions.observed_WSQS_100.to_numpy(),y.iloc[test].to_numpy(),atol=1e-8,rtol=0),
          'Frozen heldout observed WSQS differs')
    prepared=prepare_catboost(X,meta['categorical'],meta['numeric'])
    model=CatBoostRegressor()
    model.load_model(str(model_path))
    trees=int(meta['selected_iterations'])
    check(model.tree_count_==trees and 0<trees<=int(settings['catboost']['iterations']),
          'Wrong validated fixed iteration count')
    original_pred=np.asarray(model.predict(Pool(prepared.iloc[test],cat_features=meta['categorical'])),dtype=float)
    check(np.max(np.abs(original_pred-old_predictions.predicted_WSQS_100.to_numpy(dtype=float)))<1e-7,
          'Stage04 full model predictions do not reproduce')
    quartiles=quantile_settings(prepared)
    masks=evaluation_masks(prepared,test)
    pool,slices=make_scenario_pool(prepared.iloc[test],meta['categorical'],quartiles,masks)
    frozen=predict_scenarios(model,pool,slices)
    check(len(frozen)==36,'Missing frozen response scenarios')
    params=refit_params(settings,trees)

    print('='*80,flush=True)
    print('STAGE 09 | 4 SERVICE PROFILES + PRIORITY MODIFICATION + B200 DEVELOPMENT REFITS',flush=True)
    print('SOURCE SHA256:',SOURCE_SHA,flush=True)
    print('RUN SIGNATURE:',signature,flush=True)
    print('FROZEN FULL MODEL SHA256:',sha256(model_path),flush=True)
    print(f'REUSE | Stage04 full CatBoost, {trees} trees; {TEST_N} heldout; pressure eligible={int(masks["Water pressure"].sum())}',flush=True)
    print('REFIT | 200 DEVELOPMENT BOOTSTRAPS, ONE MODEL PER DRAW, NO VALIDATION RESELECTION',flush=True)
    print('='*80,flush=True)
    refits=[]
    for i in range(1,B+1):
        means,new=fit_one(i,dev,prepared,y,meta['categorical'],params,pool,slices,checkpoints,signature)
        check(len(means)==len(frozen) and set(means)==set(frozen),
              'Different scenario keys in bootstrap replicate '+str(i))
        refits.append({'replicate':i,**means})
        if i<=5 or i%10==0 or i==B:
            print(f'{"PASS" if new else "REUSE"} | Development bootstrap refit {i}/{B} | trees={trees}',flush=True)
    frames=summarize(frozen,refits,quartiles,masks,TEST_N,SOURCE_SHA,signature)
    profiles,table07,adjacent,modification,eligibility,refitdata=frames
    check((len(profiles),len(table07),len(adjacent),len(modification),len(eligibility),len(refitdata))==
          (20,4,16,4,4,B),'Missing Figure7/Table7/SI support data')
    check(all(table07['severity5_minus_1_point_wsqs_100'].between(-100,100)),
          'Contrast escaped WSQS range')
    frozen_exploratory=frozen_only_pairs(model,prepared.iloc[test],meta['categorical'],prepared)
    manifest={'status':'PASS','signature':signature,'signature_inputs':signature_inputs,
              'source_n':N,'heldout_n':TEST_N,'development_n':DEV_N,'B':B,
              'fixed_model_trees':trees,'pressure_eligible_heldout_n':int(masks['Water pressure'].sum()),
              'full_model_reused':True,'extra_frozen_only_pairs':2,
              'refit_uncertainty_scope':'development-bootstrap model-refitting conditional on fixed heldout population and selected tree count',
              'rng_scope':'explicit newly declared per-repetition RNG streams; not claimed bitwise historical',
              'interpretation':'observational model-prediction contrasts, not interventions or causal effects',
              'contains_private_refit_indices_and_CBMs':True,
              'profiles':{r['service_condition']:int(r['evaluation_n']) for r in eligibility.to_dict('records')}}
    outputs=[profiles,table07,adjacent,modification,frozen_exploratory,eligibility,refitdata]
    for filename,frame in zip(output_names[:7],outputs):
        write_csv_new(results/filename,frame)
    write_json_new(results/output_names[-1],manifest)
    result_sha={filename:sha256(results/filename) for filename in output_names}
    write_json_new(audit_path,{'status':'PASS','signature':signature,
                               'signature_inputs':signature_inputs,'source_n':N,
                               'refit_count':B,'results_dir':str(results),
                               'checkpoint_dir':str(checkpoints),
                               'result_sha256':result_sha})
    print('='*80,flush=True)
    print('PASS | Stage09 complete: four profiles, four priority modifications, all B200 refits',flush=True)
    print('PRESSURE ELIGIBLE HELDOUT N:',int(masks['Water pressure'].sum()),flush=True)
    print('RESULTS:',results,flush=True)
    print('CHECKPOINTS:',checkpoints,flush=True)
    print('AUDIT:',audit_path,flush=True)
    print('NEXT | Stage10 measurement structure, factor/reliability, weights, provider applicability',flush=True)
    print('='*80,flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('FAILED |',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        raise
