#!/usr/bin/env python3
"""Stage 07 | 1000-repetition joint-priority null for N6039 WSQS.

Original scientific *design*: complete 8-dimensional priority vectors are
jointly shuffled across fitting households and separately across validation
households, not across the test households. Outcome, service/context predictors,
original partitions and held-out priority values remain unchanged. The null
full SCP model undergoes validation-selected iteration and refitting on the
whole (permuted) development sample before evaluation on untouched test rows.
The frozen SC model from Stage04 is the comparator on the identical test rows.

The historical 100-repetition source loop is not included in the portable
October 19 ZIP. This extension implements the previously verified original
null DESIGN; all exact pseudorandom seed and call conventions are explicitly
reported in the Stage07 audit, rather than claiming bitwise equivalence to an
unavailable historical private script. No SHAP-based null is performed.

Run using the locked scientific environment:
 WORK="$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_WSQS_RERUN_20261030_N6039"
 "$WORK/.venv_wsqs_n6039/bin/python" ~/Downloads/WSQS_STAGE07_PRIORITY_NULL_B1000_N6039.py

--self-test verifies permutation and inference logic without accessing study data.
--max-new N may be used to run at most N NEW null repetitions and exit cleanly;
repeat the same default command to finish. Never use this option for final audit.

The run writes only per-repetition aggregate JSON files and final aggregate CSVs;
no 1000 redundant model binaries or ResponseId-level predictions are exported.
Existing scientific sources, Stage04/05/06 models and historical input unchanged.
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

PROJECT='NPJ_WSQS_RERUN_20261030_N6039'
STAGE='STAGE07_PRIORITY_NULL_B1000_N6039'
B=1000
SEED=20260714
SEED_OFFSET=1000
# Historical independent fit/validation permutations have been verified;
# we document separate streams to eliminate PRNG-order ambiguities.
VALIDATION_STREAM_OFFSET=100000


def require(ok, why):
    if not ok: raise RuntimeError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fp:
        for chunk in iter(lambda:fp.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_once_json(path, data):
    path=Path(path)
    require(not path.exists(),'Existing result must not be overwritten: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False,encoding='utf8',suffix='.partial') as fp:
        tmp=Path(fp.name)
        json.dump(data,fp,indent=2,allow_nan=False)
        fp.write('\n')
    try:
        require(not path.exists(),'Result collision: '+str(path))
        os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def write_once_csv(path, frame):
    path=Path(path)
    require(not path.exists(),'Existing aggregate must not be overwritten: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False,encoding='utf8',suffix='.partial') as fp:
        tmp=Path(fp.name)
        frame.to_csv(fp,index=False,float_format='%.17g')
    try:
        require(not path.exists(),'Result collision: '+str(path))
        os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def shuffle_priority_vectors(X, p_cols, indices, seed):
    """Original conceptual method, with an independent seeded stream per subset.

    Assign complete P vectors, not eight independently permuted columns.
    Never change households outside 'indices'. X must have canonical RangeIndex.
    """
    require(len(p_cols)==8 and len(set(p_cols))==8,'Must jointly shuffle exactly eight priority columns')
    idx=np.asarray(indices,dtype=int)
    require(len(np.unique(idx))==len(idx) and np.all(idx>=0) and np.all(idx<len(X)),
            'Invalid permutation indices')
    result=X.copy(deep=True)
    order=np.random.default_rng(int(seed)).permutation(len(idx))
    values=X.iloc[idx][p_cols].to_numpy(copy=True)
    result.loc[result.index[idx],p_cols]=values[order]
    return result


def shuffle_fit_validation(X, priority_cols, fit, val, test, base_seed):
    require(not(set(fit)&set(val) or set(fit)&set(test) or set(val)&set(test)),
            'Training and heldout subsets overlap')
    randomized=shuffle_priority_vectors(X,priority_cols,fit,base_seed)
    randomized=shuffle_priority_vectors(randomized,priority_cols,val,
                                        base_seed+VALIDATION_STREAM_OFFSET)
    require(np.array_equal(randomized.iloc[test][priority_cols].to_numpy(),
                           X.iloc[test][priority_cols].to_numpy()),
            'Randomization touched heldout priority records')
    non_priority=[c for c in X if c not in priority_cols]
    require(randomized[non_priority].equals(X[non_priority]),
            'Randomization changed non-priority predictors')
    return randomized


def fit_null_once(Xnull, y, cats, fit, val, test, dev, settings):
    from wsqs.models import prepare_catboost,summary_metrics
    prepared=prepare_catboost(Xnull,cats,[c for c in Xnull if c not in cats])
    params=dict(settings['catboost'])
    params.update(verbose=False,allow_writing_files=False)
    first=CatBoostRegressor(**params)
    first.fit(Pool(prepared.iloc[fit],y.iloc[fit],cat_features=cats),
              eval_set=Pool(prepared.iloc[val],y.iloc[val],cat_features=cats),
              use_best_model=True)
    selected=first.get_best_iteration()
    require(selected is not None and int(selected)>=0,'No validation-selected boosting iteration')
    n_trees=int(selected)+1
    last_params=dict(params)
    last_params['iterations']=n_trees
    last_params.pop('od_type',None)
    last_params.pop('od_wait',None)
    final=CatBoostRegressor(**last_params)
    final.fit(Pool(prepared.iloc[dev],y.iloc[dev],cat_features=cats))
    pred=np.asarray(final.predict(Pool(prepared.iloc[test],cat_features=cats)),dtype=float)
    require(pred.shape==(len(test),) and np.isfinite(pred).all(),'Invalid null heldout predictions')
    metrics=summary_metrics(y.iloc[test],pred)
    return n_trees,metrics


def mc_p(observed, null_deltas):
    a=np.asarray(null_deltas,dtype=float)
    require(a.ndim==1 and len(a)>0 and np.isfinite(a).all(),'Bad null statistic vector')
    exceed=int(np.count_nonzero(a>=observed))
    return exceed, (1+exceed)/(len(a)+1)


def self_test():
    from sklearn.metrics import r2_score
    rng=np.random.default_rng(81)
    X=pd.DataFrame(rng.normal(size=(240,6)),columns=[f's{i}' for i in range(6)])
    for j in range(8): X[f'p{j}']=np.arange(240)+j*1000
    pcols=[f'p{j}' for j in range(8)]
    fit=np.arange(0,150); val=np.arange(150,195); test=np.arange(195,240)
    shuffled=shuffle_fit_validation(X,pcols,fit,val,test,20261714)
    require(shuffled.iloc[test].equals(X.iloc[test]),'Heldout rows altered')
    require(shuffled[[f's{i}' for i in range(6)]].equals(X[[f's{i}' for i in range(6)]]),
            'Other features altered')
    for ids in (fit,val):
        expected={tuple(row) for row in X.iloc[ids][pcols].to_numpy()}
        actual={tuple(row) for row in shuffled.iloc[ids][pcols].to_numpy()}
        require(expected==actual,'Joint priority vectors changed')
    require(not shuffled.iloc[fit][pcols].equals(X.iloc[fit][pcols]),'Shuffle unexpectedly identity')
    require(shuffle_fit_validation(X,pcols,fit,val,test,20261714).equals(shuffled),
            'RNG not deterministic')
    require(mc_p(0.2,[0.1,.2,.3])==(2,.75),'Plus-one tail arithmetic wrong')
    # Lightweight end-to-end fit on synthetic data only; no real cohort required.
    first=CatBoostRegressor(iterations=12,depth=3,learning_rate=.08,random_seed=12,
                            verbose=False,allow_writing_files=False)
    y=rng.normal(size=len(X))
    first.fit(X.iloc[fit],y[fit],eval_set=(X.iloc[val],y[val]),use_best_model=True)
    require(np.isfinite(r2_score(y[test],first.predict(X.iloc[test]))),'Synthetic model invalid')
    print('PASS | Joint-vector fit/validation-only randomization preserved all 8D records')
    print('PASS | Untouched heldout records, non-priority predictors, deterministic seeds')
    print('PASS | Monte Carlo upper-tail formula and synthetic CatBoost execution')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,default=Path.home()/'Desktop/Spring 2026/Chapter 3 MANUAL RERUN')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--max-new',type=int,default=0,help='Optional cap on NEW repetitions in current invocation')
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    require(args.max_new>=0,'--max-new must be nonnegative')
    os.umask(0o077)
    work=args.base.expanduser().resolve()/PROJECT
    venv=work/'.venv_wsqs_n6039/bin/python'
    require(venv.exists() and Path(sys.executable).resolve()==venv.resolve(),
            'Use locked project venv Python: '+str(venv))
    source=work/'00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    freeze=work/'07_audit/COHORT_DATASET_LOCK_N6039.json'
    env=work/'07_audit/SCIENTIFIC_ENVIRONMENT_N6039.json'
    stage04=work/'07_audit/STAGE04_PRIMARY_CROSSDOMAIN_N6039_COMPLETE.json'
    stage06=work/'07_audit/STAGE06_REPEATED_SPLITS_BENCHMARK_N6039_COMPLETE.json'
    ck04=work/'03_run/checkpoints/STAGE04_PRIMARY_CROSSDOMAIN_N6039'
    code=work/'02_code_private/reconciled/WSQS_Study_Repro_N6039'
    required=[source,freeze,env,stage04,stage06,
             ck04/'BLOCK_SC/model_manifest.json',ck04/'BLOCK_SCP/model_manifest.json',
             ck04/'BLOCK_SC/heldout_predictions.csv',ck04/'BLOCK_SCP/heldout_predictions.csv',
             code/'src/wsqs/models.py',code/'src/wsqs/features.py',code/'src/wsqs/extended.py',
             code/'config/models.json',code/'config/scientific_plans.json',
             code/'config/feature_spec.json']
    for item in required: require(item.is_file(),'Missing prerequisite: '+str(item))
    data_sha=sha(source)
    lock,a04,a06=read_json(freeze),read_json(stage04),read_json(stage06)
    require(data_sha==lock['output_sha256']['primary']==a04['fingerprint']['source_cohort_sha256']
            ==a06['inputs']['dataset_sha256'],'Source/frozen SHA mismatch')
    require(a04['status']=='PASS' and a06['status']=='PASS','Prior stages must pass')
    sys.path.insert(0,str(code/'src'))
    from wsqs.features import construct_features
    from wsqs.extended import DOMAINS,scientific_plans
    from wsqs.models import model_settings,partition_indices,summary_metrics
    settings=model_settings();plans=scientific_plans()
    require(settings['primary_seed']==SEED and settings['catboost']['random_seed']==SEED,
            'Model seeds changed')
    require(len(plans['full'])==33 and len(plans['blocks']['P'])==8 and
            len(plans['blocks']['S'])==22 and len(plans['blocks']['C'])==3,'Predictor plan changed')
    fingerprints={str(p.relative_to(work)):sha(p) for p in required[-6:]}
    inputs={'data_sha256':data_sha,'stage04_signature':a04['signature'],
            'stage06_signature':a06['signature'],'stage07_script_sha256':sha(Path(__file__).resolve()),
            'source_sha256':fingerprints,'B':B,'seed':SEED,'seed_offset':SEED_OFFSET,
            'validation_stream_offset':VALIDATION_STREAM_OFFSET,
            'randomization':'joint eight-dimensional priority vectors, separately within fitting and validation',
            'heldout_shuffle':False,'model_selection':'validation-selected trees, then full development refit',
            'reference_model':'frozen Stage04 SC'}
    signature=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    checkpoint=work/'03_run/checkpoints'/STAGE
    results=work/'04_numerical_results/inference'/STAGE
    audit=work/'07_audit'/(STAGE+'_COMPLETE.json')
    if audit.exists():
        prior=read_json(audit)
        require(prior['signature']==signature,'Stage07 signature changed')
        for relative,h in prior['output_sha256'].items():
            require(sha(work/relative)==h,'Stage07 audited result changed: '+relative)
        print('PASS | Stage07 already complete; all aggregate outputs verified');return
    for folder in (checkpoint,results):folder.mkdir(parents=True,exist_ok=True)

    frame=pd.read_csv(source,low_memory=False,dtype={'ResponseId':str})
    require(frame.shape==(6039,182) and frame['ResponseId'].is_unique and
            frame['ResponseId'].notna().all(),'Locked dataset dimensions/IDs invalid')
    if 'municipal_supply_applicable' not in frame:
        require('source09M_municipal_eligible' in frame,'Missing provider field')
        frame=frame.rename(columns={'source09M_municipal_eligible':'municipal_supply_applicable'})
    X,cats,nums=construct_features(frame)
    require(X.shape==(6039,33) and list(X.columns)==plans['full'] and
            all(p in nums for p in plans['blocks']['P']),'33-column predictor plan differs')
    sev=frame[[f'{d}_raw_severity' for d in DOMAINS]].apply(pd.to_numeric,errors='coerce')
    require(sev.notna().sum(axis=1).between(7,8).all(),'CORE8 completeness error')
    y=pd.Series((5.0-sev.mean(axis=1))*25.0,dtype=float)
    require(y.between(0,100).all() and y.notna().all(),'CORE8 outcome invalid')
    fit,val,test,dev=partition_indices(len(frame),SEED)
    require((len(fit),len(val),len(test),len(dev))==(3864,967,1208,4831),'Model partition mismatch')

    metas={x:read_json(ck04/('BLOCK_'+x)/'model_manifest.json') for x in ('SC','SCP')}
    pred={x:pd.read_csv(ck04/('BLOCK_'+x)/'heldout_predictions.csv',dtype={'ResponseId':str})
          for x in ('SC','SCP')}
    for x in ('SC','SCP'):
        mf=metas[x]; pt=pred[x]
        require(mf['signature']==a04['signature'],'Stage04 checkpoint signature mismatch '+x)
        require(mf['features']==[c for c in plans['full'] if c in set(plans['blocks']['S']+plans['blocks']['C']+(plans['blocks']['P'] if x=='SCP' else []))],
                'Reference model feature plan changed '+x)
        for name,h in mf['output_sha256'].items():
            require(sha(ck04/('BLOCK_'+x)/name)==h,'Stage04 checkpoint corrupted '+x+'/'+name)
        require(pt['canonical_row_index'].astype(int).tolist()==test.tolist() and
                pt['ResponseId'].tolist()==frame['ResponseId'].iloc[test].astype(str).tolist(),
                'Stage04 test members changed '+x)
        require(np.allclose(pt['observed_WSQS_100'].astype(float),y.iloc[test].to_numpy(),atol=1e-10,rtol=0),
                'Stage04 target values changed '+x)
        observed_metrics=summary_metrics(y.iloc[test],pt['predicted_WSQS_100'].astype(float))
        require(np.isclose(observed_metrics['r2'],mf['metrics']['r2'],atol=1e-11,rtol=0),
                'Stage04 reference R2 inconsistent '+x)
    sc_r2=float(metas['SC']['metrics']['r2'])
    scp_r2=float(metas['SCP']['metrics']['r2'])
    observed_gain=scp_r2-sc_r2
    require(np.isclose(observed_gain,.006275191,atol=2e-8,rtol=0),
            'Observed N6039 SC->SCP gain unexpected; investigate before B1000')

    print('='*78)
    print('STAGE07 | PRIORITY NULL B1000 | joint priority vectors, training-only')
    print('SOURCE SHA256:',data_sha)
    print('RUN SIGNATURE:',signature)
    print(f'FROZEN OBSERVED | SC R2={sc_r2:.9f} | SCP R2={scp_r2:.9f} | delta={observed_gain:.9f}')
    print('TRAIN/VAL/TEST: 3864 / 967 / 1208; test priority columns never shuffled')
    print('NO REFIT | frozen original SC and SCP reference models')
    print('='*78,flush=True)

    pcols=plans['blocks']['P']; completed=0; added=0
    for i in range(B):
        key=f'NULL_{i+1:04d}'
        path=checkpoint/(key+'.json')
        seed_fit=SEED+SEED_OFFSET+i
        seed_val=seed_fit+VALIDATION_STREAM_OFFSET
        if path.exists():
            entry=read_json(path)
            require(entry['signature']==signature and entry['replicate_index']==i+1 and
                    entry['fit_shuffle_seed']==seed_fit and entry['validation_shuffle_seed']==seed_val,
                    'Null checkpoint has stale settings: '+str(path))
            require(np.isfinite(entry['null_delta_r2']),'Null checkpoint is not finite: '+str(path))
            completed+=1
            continue
        if args.max_new and added>=args.max_new:break
        randomized=shuffle_fit_validation(X,pcols,fit,val,test,seed_fit)
        tree_count,metrics=fit_null_once(randomized,y,cats,fit,val,test,dev,settings)
        null_r2=float(metrics['r2'])
        result={'signature':signature,'replicate_index':i+1,
                'fit_shuffle_seed':seed_fit,'validation_shuffle_seed':seed_val,
                'selected_iterations':tree_count,
                'SC_reference_heldout_r2':sc_r2,
                'null_SCP_heldout_r2':null_r2,
                'null_delta_r2':null_r2-sc_r2,
                'null_rmse_100':float(metrics['rmse_100']),
                'null_mae_100':float(metrics['mae_100']),
                'randomization_scope':'complete priority vectors separately in fitting and validation only; test unchanged',
                'test_households':len(test)}
        write_once_json(path,result)
        completed+=1;added+=1
        print(f'PASS | {key} | null delta R2={result["null_delta_r2"]:+.9f} | '
              f'trees={tree_count} | {completed}/{B}',flush=True)

    if completed<B:
        print(f'CHECKPOINTED | {completed}/{B} valid repetitions; run same command to resume')
        return
    rows=[read_json(checkpoint/(f'NULL_{i:04d}.json')) for i in range(1,B+1)]
    require(len(rows)==B and [r['replicate_index'] for r in rows]==list(range(1,B+1)),
            'Incomplete or unordered priority-null evidence')
    null=np.array([r['null_delta_r2'] for r in rows],dtype=float)
    exceed,pval=mc_p(observed_gain,null)
    summary={'n_households':6039,'n_heldout':len(test),'B':B,
             'observed_SC_r2':sc_r2,'observed_SCP_r2':scp_r2,
             'observed_incremental_r2':observed_gain,'null_mean_incremental_r2':float(null.mean()),
             'null_sd_incremental_r2':float(null.std(ddof=1)),
             'null_2p5_percentile':float(np.percentile(null,2.5)),
             'null_50_percentile':float(np.percentile(null,50)),
             'null_97p5_percentile':float(np.percentile(null,97.5)),
             'null_min':float(null.min()),'null_max':float(null.max()),
             'null_greater_or_equal_observed':exceed,
             'empirical_upper_tail_p_plus_one':float(pval),
             'monte_carlo_p_floor':float(1/(B+1)),
             'interpretation_boundary':'reference randomization of priority information, not causal effect or SHAP significance'}
    long_rows=[{k:v for k,v in row.items() if k in
                ('replicate_index','fit_shuffle_seed','validation_shuffle_seed','selected_iterations',
                 'SC_reference_heldout_r2','null_SCP_heldout_r2','null_delta_r2',
                 'null_rmse_100','null_mae_100')}
               for row in rows]
    dest1=results/'01_priority_null_B1000_repetitions.csv'
    dest2=results/'02_priority_null_B1000_summary.csv'
    dest3=results/'03_priority_null_B1000_method.json'
    write_once_csv(dest1,pd.DataFrame(long_rows))
    write_once_csv(dest2,pd.DataFrame([summary]))
    write_once_json(dest3,{'inputs':inputs,'method':'train-only joint eight-dimensional vector shuffle; '
                                'validation-selected CatBoost followed by full development refit',
                           'summary':summary,
                           'historical_method_note':'Randomization design recovered from prior verified research record. '
                                   'Exact old private B100 pseudorandom seed conventions were not independently audited in this environment; '
                                   'all B1000 seed streams here are disclosed explicitly.'})
    hashes={str(p.relative_to(work)):sha(p) for p in (dest1,dest2,dest3)}
    write_once_json(audit,{'status':'PASS','signature':signature,'inputs':inputs,
                           'summary':summary,'output_sha256':hashes,
                           'checkpoints':B,'private_checkpoint_root':str(checkpoint)})
    print('='*78)
    print(f'PASS | Stage07 completed all {B} priority-null model refits')
    print(f'OBSERVED DELTA R2: {observed_gain:.9f} | NULL MEAN: {null.mean():.9f}')
    print(f'NULL 97.5%: {np.percentile(null,97.5):.9f} | EXCEED: {exceed}/{B} | PLUS-ONE P: {pval:.9f}')
    print('RESULTS:',results)
    print('AUDIT:',audit)
    print('NEXT | Grouped SHAP and uncertainty; 200-refit service response; measurement/provider checks')
    print('='*78,flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('FAILED |',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        raise
