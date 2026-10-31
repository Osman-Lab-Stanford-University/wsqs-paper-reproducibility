#!/usr/bin/env python3
"""WSQS Stage 06 | Ten matched block-model splits and seven-learner benchmark.

RUN AFTER Stage 04 and Stage 05, using the project's isolated Python:
  WORK="$HOME/Desktop/Spring 2026/Chapter 3 MANUAL RERUN/NPJ_WSQS_RERUN_20261030_N6039"
  "$WORK/.venv_wsqs_n6039/bin/python" \
    ~/Downloads/WSQS_STAGE06_REPEATED_SPLITS_BENCHMARK_N6039.py

The ten seeds, learner configurations, partitions, feature engineering and two-
pass CatBoost fitting procedure come from the archived Oct 19 source.
Reuses Stage 04 seed-20260714 CBM models; fits 63 *new* block models, and 50
new non-CatBoost learner fits (Mean baseline needs no fit). Includes Stage 04's
first CatBoost full model in the 70-row learner benchmark.

All respondent-linked predictions remain PRIVATE. Result CSVs are numerical
sources for SI S16/S17/S19 and the repeated-split parts of Figure 8.
No experimental models or previous results are overwritten. Resume is safe.

IMPORTANT: Ten overlapping random partitions are sensitivity checks, not ten
independent replications; report descriptive variation without treating them as
independent sampling units for inferential confidence intervals.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = 'NPJ_WSQS_RERUN_20261030_N6039'
STAGE = 'STAGE06_REPEATED_SPLITS_BENCHMARK_N6039'
STAGE04 = 'STAGE04_PRIMARY_CROSSDOMAIN_N6039'
SEED0 = 20260714
BLOCKS = ('S','C','P','SC','SP','CP','SCP')


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once_json(path, payload):
    require(not path.exists(), 'Refusing to overwrite: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf8', dir=path.parent, delete=False, suffix='.partial') as stream:
        part = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(part, path)


def write_once_csv(path, table):
    # A crash between aggregate CSV production and final audit can be resumed.
    # Compare full output bytes before reusing; never silently replace a result.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf8', dir=path.parent, delete=False, suffix='.partial') as stream:
        part = Path(stream.name)
        table.to_csv(stream, index=False, float_format='%.17g')
    try:
        if path.exists():
            require(sha(path) == sha(part), 'Existing aggregate conflicts with recomputation: ' + str(path))
            print('REUSE | identical aggregate:', path.name, flush=True)
        else:
            os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()


def import_stage04(script_path):
    require(script_path.is_file(), 'Missing verified Stage04 runner: ' + str(script_path))
    spec = importlib.util.spec_from_file_location('wsqs_verified_stage04', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_block(stage04_checkpoints, stage04_signature, name, seed, fit, val, test, dev, columns, ids):
    """Reuse existing first-seed fit; verify checksum, prediction IDs and metrics."""
    assert seed == SEED0
    folder = stage04_checkpoints / ('BLOCK_' + name)
    mf = folder / 'model_manifest.json'
    require(mf.is_file(), 'Stage04 model checkpoint missing: ' + str(mf))
    meta = load_json(mf)
    require(meta['signature'] == stage04_signature, 'Stage04 model fingerprint mismatch')
    require(meta['features'] == columns, f'Stage04 feature membership mismatch: {name}')
    require((meta['fitting_n'],meta['validation_n'],meta['development_n'],meta['heldout_n']) ==
            (len(fit),len(val),len(dev),len(test)), f'Stage04 partition mismatch: {name}')
    for filename, expected in meta['output_sha256'].items():
        path = folder / filename
        require(path.is_file() and sha(path) == expected, f'Stage04 checkpoint corrupted: {name}/{filename}')
    pred = pd.read_csv(folder/'heldout_predictions.csv', dtype={'ResponseId':str})
    require(pred['canonical_row_index'].to_numpy(dtype=int).tolist() == test.astype(int).tolist(),
            f'Stage04 ordered test row mismatch: {name}')
    require(pred['ResponseId'].tolist() == ids.iloc[test].astype(str).tolist(),
            f'Stage04 test ResponseIds differ: {name}')
    return meta


def learner_fit(name, seed, fit, val, test, dev, X_skl, target,
                cats, nums, ids, config, root, signature, metrics_function, model_function):
    """Reproduce portable source other_predictions; private checksum checkpoint."""
    key = str(seed) + '_' + name.replace(' ', '_')
    folder = root / key
    if folder.exists():
        mf = load_json(folder / 'model_manifest.json')
        require(mf['signature'] == signature and mf['learner'] == name and mf['seed'] == seed,
                'Stale learner checkpoint: ' + key)
        require((mf['fitting_n'], mf['validation_n'], mf['development_n'], mf['heldout_n']) ==
                (len(fit),len(val),len(dev),len(test)), 'Checkpoint partition size changed: '+key)
        for filename, digest in mf['output_sha256'].items():
            require(sha(folder / filename) == digest, 'Invalid learner output SHA: '+key+'/'+filename)
        pred = pd.read_csv(folder / 'heldout_predictions.csv', dtype={'ResponseId':str})
        require(pred['canonical_row_index'].to_numpy(dtype=int).tolist() == test.astype(int).tolist() and
                pred['ResponseId'].tolist() == ids.iloc[test].astype(str).tolist(),
                'Learner test membership mismatch: '+key)
        print(f'REUSE | {seed} | {name} | R2={mf["metrics"]["r2"]:.9f}', flush=True)
        return mf

    predictions, selected_alpha = model_function(name, X_skl, target, cats, nums,
                                                  seed, fit, val, test, dev, config)
    predictions = np.asarray(predictions, dtype=float)
    require(predictions.shape == (len(test),) and np.isfinite(predictions).all(), 'Bad '+name+' predictions')
    metrics = metrics_function(target.iloc[test], predictions)
    # The mean-only baseline has undefined Spearman rho. JSON uses null, not NaN.
    metrics = {k:(float(v) if np.isfinite(v) else None) for k,v in metrics.items()}
    selected = float(selected_alpha) if np.isfinite(selected_alpha) else None
    private = pd.DataFrame({
        'canonical_row_index': test.astype(int),
        'ResponseId': ids.iloc[test].astype(str).to_numpy(),
        'observed_WSQS_100': target.iloc[test].to_numpy(dtype=float),
        'predicted_WSQS_100': predictions,
        'residual_observed_minus_predicted': target.iloc[test].to_numpy(dtype=float) - predictions,
    })
    tmp = Path(tempfile.mkdtemp(prefix=key+'.partial.', dir=root))
    try:
        private.to_csv(tmp / 'heldout_predictions.csv', index=False, float_format='%.17g')
        mf = {'learner':name,'seed':seed,'signature':signature,'fitting_n':len(fit),
              'validation_n':len(val),'development_n':len(dev),'heldout_n':len(test),
              'selected_alpha':selected,'selected_iterations':None,'metrics':metrics,
              'output_sha256':{'heldout_predictions.csv':sha(tmp/'heldout_predictions.csv')},
              'private_respondent_outputs':True,
              'method':'wsqs.models.other_predictions; fixed configuration or validation-selected Ridge alpha'}
        (tmp/'model_manifest.json').write_text(json.dumps(mf, indent=2, allow_nan=False)+'\n',encoding='utf8')
        os.replace(tmp,folder)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(f'PASS | {seed} | {name} | R2={metrics["r2"]:.9f} | RMSE={metrics["rmse_100"]:.9f}', flush=True)
    return mf


def model_metrics_row(seed, name, meta, n_features, split):
    return {'seed':seed,'block':name,'n_predictors':n_features,'selected_iterations':meta.get('selected_iterations'),
            'fitting_n':len(split[0]),'validation_n':len(split[1]),
            'development_n':len(split[3]),'heldout_n':len(split[2]),**meta['metrics']}


def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,default=Path.home()/'Desktop/Spring 2026/Chapter 3 MANUAL RERUN')
    ap.add_argument('--stage04-script',type=Path,default=Path.home()/'Downloads/WSQS_STAGE04_PRIMARY_CROSSDOMAIN_N6039.py')
    args=ap.parse_args()
    work=args.base.expanduser().resolve()/PROJECT
    venv=work/'.venv_wsqs_n6039/bin/python'
    require(Path(sys.executable).resolve()==venv.resolve(), 'Use isolated WSQS Python: '+str(venv))
    source=work/'00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    stage04_audit=work/'07_audit/STAGE04_PRIMARY_CROSSDOMAIN_N6039_COMPLETE.json'
    stage05_audit=work/'07_audit/STAGE05_SENSITIVITY_SATISFACTION_N6039_COMPLETE.json'
    freeze=work/'07_audit/COHORT_DATASET_LOCK_N6039.json'
    code=work/'02_code_private/reconciled/WSQS_Study_Repro_N6039'
    model_root=work/'03_run/checkpoints'/STAGE
    results_root=work/'04_numerical_results/modeling'/STAGE
    result_audit=work/'07_audit'/(STAGE+'_COMPLETE.json')
    script04=args.stage04_script.expanduser().resolve()
    for needed in (source,stage04_audit,stage05_audit,freeze,script04,
                   code/'src/wsqs/models.py',code/'src/wsqs/features.py',
                   code/'src/wsqs/extended.py',code/'config/models.json',
                   code/'config/scientific_plans.json',code/'config/feature_spec.json'):
        require(needed.is_file(),'Missing prerequisite: '+str(needed))
    digest_source=sha(source)
    locked=load_json(freeze)
    audit04=load_json(stage04_audit)
    audit05=load_json(stage05_audit)
    require(digest_source==locked['output_sha256']['primary']==audit04['fingerprint']['source_cohort_sha256']
            ==audit05['inputs']['dataset_sha256'], 'Source cohort SHA disagrees with prior stages')
    require(audit04['status']=='PASS' and audit05['status']=='PASS', 'Stage04/05 have not passed')

    sys.path.insert(0,str(code/'src'))
    from wsqs.extended import BLOCK_KEYS,block_columns,scientific_plans
    from wsqs.features import construct_features
    from wsqs.models import (LEARNERS,model_settings,partition_indices,summary_metrics,
                             prepare_sklearn,other_predictions)
    module=import_stage04(script04)
    require(tuple(BLOCK_KEYS)==BLOCKS and tuple(LEARNERS)==
            ('CatBoost','HistGradientBoosting','Extra Trees','Random Forest','ElasticNet','Ridge','Mean baseline'),
            'Historic block or benchmark learner definitions changed')
    settings=model_settings()
    seeds=[int(v) for v in settings['repeated_split_seeds']]
    require(seeds==list(range(20260714,20260724)), 'Historic ten split seeds changed')
    plans=scientific_plans()
    require(len(plans['full'])==33 and len(plans['blocks']['S'])==22 and
            len(plans['blocks']['C'])==3 and len(plans['blocks']['P'])==8, 'Feature plan mismatch')
    fingerprints={p.name:sha(p) for p in (code/'src/wsqs/models.py',code/'src/wsqs/features.py',
                                         code/'src/wsqs/extended.py',code/'config/models.json',
                                         code/'config/scientific_plans.json',code/'config/feature_spec.json')}
    inputs={'dataset_sha256':digest_source,'stage04_signature':audit04['signature'],
            'stage05_signature':audit05['signature'],'stage04_runner_sha256':sha(script04),
            'stage06_runner_sha256':sha(Path(__file__).resolve()),'scientific_source_sha256':fingerprints,
            'seeds':seeds}
    signature=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    print('='*78)
    print('STAGE 06 | TEN MATCHED BLOCK SPLITS + SEVEN LEARNER BENCHMARK')
    print('SOURCE SHA256:',digest_source)
    print('RUN SIGNATURE:',signature)
    print('EXISTING | Stage04 seed 20260714 models will be reused; not refitted')
    print('='*78,flush=True)

    if result_audit.exists():
        prior=load_json(result_audit)
        require(prior['signature']==signature,'Existing result fingerprint differs')
        for relative,digest in prior['aggregate_output_sha256'].items():
            require(sha(work/relative)==digest,'Corrupt completed aggregate '+relative)
        print('PASS | Stage06 previously completed; existing output hashes verified')
        return

    df=pd.read_csv(source,dtype={'ResponseId':str},low_memory=False)
    require(df.shape==(6039,182) and df['ResponseId'].notna().all() and df['ResponseId'].is_unique,
            'Dataset dimensions/ResponseIds changed')
    if 'municipal_supply_applicable' not in df:
        require('source09M_municipal_eligible' in df, 'Provider eligibility column missing')
        df=df.rename(columns={'source09M_municipal_eligible':'municipal_supply_applicable'})
    X,cats,nums=construct_features(df)
    require(X.shape==(6039,33) and list(X.columns)==plans['full'] and
            len(cats)+len(nums)==33,'Original 33-predictor model specification changed')
    from wsqs.extended import domain_severities
    raw=domain_severities(df)
    y=pd.Series((5.0-raw.mean(axis=1))*25.0,dtype=float)
    require(np.isfinite(y).all() and y.between(0,100).all(),'Invalid CORE8 outcome')
    X_skl=prepare_sklearn(X,cats,nums)
    model_root.mkdir(parents=True,exist_ok=True)
    results_root.mkdir(parents=True,exist_ok=True)
    first_seed_models=work/'03_run/checkpoints'/STAGE04
    rows_block=[]
    rows_benchmark=[]
    rows_pair=[]
    member_summary=[]
    for repeat,seed in enumerate(seeds,1):
        fit,val,test,dev=partition_indices(6039,seed)
        require((len(fit),len(val),len(test),len(dev))==(3864,967,1208,4831),
                'Partition count changed for seed '+str(seed))
        partition=(fit,val,test,dev)
        print('\n'+'-'*76)
        print(f'SPLIT {repeat}/10 | seed={seed} | 7 CatBoost blocks + 6 other benchmark learners')
        print('-'*76,flush=True)
        metrics_by_block={}
        used_stage04=0
        for block in BLOCKS:
            cols=block_columns(block,plans)
            if seed==SEED0:
                mf=verified_block(first_seed_models,audit04['signature'],block,seed,fit,val,test,dev,
                                  cols,df['ResponseId'])
                print(f'REUSE | {seed} | BLOCK_{block} | R2={mf["metrics"]["r2"]:.9f}',flush=True)
                used_stage04+=1
            else:
                mf=module.train_and_record(str(seed)+'_BLOCK_'+block,cols,y,fit,val,test,dev,
                                           X,cats,nums,df['ResponseId'],settings,model_root,signature)
            metrics_by_block[block]=mf
            rows_block.append(model_metrics_row(seed,block,mf,len(cols),partition))
            if block=='SCP':
                rows_benchmark.append({'seed':seed,'learner':'CatBoost','selected_iterations':mf['selected_iterations'],
                                       'selected_alpha':None,'heldout_n':len(test),**mf['metrics']})
        require(used_stage04 in (0,7), 'Partial first-seed reuse is invalid')
        baseline=np.full(len(test),float(y.iloc[dev].mean()))
        base_metrics=summary_metrics(y.iloc[test],baseline)
        rows_block.append({'seed':seed,'block':'Mean baseline','n_predictors':0,'selected_iterations':None,
                           'fitting_n':len(fit),'validation_n':len(val),'development_n':len(dev),
                           'heldout_n':len(test),**base_metrics})
        for learner in LEARNERS[1:]:
            mf=learner_fit(learner,seed,fit,val,test,dev,X_skl,y,cats,nums,df['ResponseId'],settings,
                           model_root,signature,summary_metrics,other_predictions)
            rows_benchmark.append({'seed':seed,'learner':learner,'selected_iterations':None,
                                   'selected_alpha':mf['selected_alpha'],'heldout_n':len(test),**mf['metrics']})
        pair={'seed':seed,'heldout_n':len(test)}
        for left,right in [('S','SP'),('SC','SCP'),('S','SC'),('SP','SCP')]:
            pair[f'{right}_minus_{left}_RMSE_100']=metrics_by_block[right]['metrics']['rmse_100']-metrics_by_block[left]['metrics']['rmse_100']
            pair[f'{right}_minus_{left}_R2']=metrics_by_block[right]['metrics']['r2']-metrics_by_block[left]['metrics']['r2']
        rows_pair.append(pair)
        member_summary.append({'seed':seed,'fit_n':len(fit),'val_n':len(val),'dev_n':len(dev),'test_n':len(test),
                               'heldout_ResponseId_order_sha256':hashlib.sha256('\n'.join(df['ResponseId'].iloc[test].astype(str)).encode()).hexdigest(),
                               'reused_stage04_models':used_stage04})
        print(f'PASS | Completed split {repeat}/10 | full model R2={metrics_by_block["SCP"]["metrics"]["r2"]:.8f}',flush=True)

    all_blocks=pd.DataFrame(rows_block)
    all_bench=pd.DataFrame(rows_benchmark)
    all_pair=pd.DataFrame(rows_pair)
    require(all_blocks.shape[0]==80 and all_bench.shape[0]==70 and all_pair.shape[0]==10,
            'Final repeated-split row counts do not match original design')
    require(all_blocks[['seed','block']].drop_duplicates().shape[0]==80,'Repeated block ID duplicates')
    require(all_bench[['seed','learner']].drop_duplicates().shape[0]==70,'Benchmark ID duplicates')
    means=(all_blocks.groupby('block',sort=False).agg(n_splits=('seed','size'),
            mean_rmse_100=('rmse_100','mean'),sd_rmse_100=('rmse_100','std'),
            mean_mae_100=('mae_100','mean'),mean_r2=('r2','mean'),sd_r2=('r2','std'),
            mean_spearman=('spearman','mean')).reset_index())
    means=means.set_index('block').loc[list(BLOCKS)+['Mean baseline']].reset_index()
    bench_summary=(all_bench.groupby('learner',sort=False).agg(n_splits=('seed','size'),
            mean_rmse_100=('rmse_100','mean'),sd_rmse_100=('rmse_100','std'),
            mean_rmse_raw_equivalent=('rmse_raw_equivalent','mean'),
            sd_rmse_raw_equivalent=('rmse_raw_equivalent','std'),
            mean_mae_100=('mae_100','mean'),mean_r2=('r2','mean'),
            mean_spearman=('spearman','mean')).reset_index())
    bench_summary=bench_summary.set_index('learner').loc[list(LEARNERS)].reset_index()
    paired_summary=pd.DataFrame([{'contrast':name,'n_splits':10,
                                  'mean_change':float(all_pair[name].mean()),
                                  'sd_change':float(all_pair[name].std(ddof=1)),
                                  'positive_splits':int((all_pair[name]>0).sum()),
                                  'negative_splits':int((all_pair[name]<0).sum())}
                                 for name in all_pair.columns if name not in ('seed','heldout_n')])
    outputs={
        '01_ten_split_all_block_metrics.csv':all_blocks,
        '02_ten_split_block_summary.csv':means,
        '03_ten_split_learner_benchmark_metrics.csv':all_bench,
        '04_ten_split_learner_benchmark_summary.csv':bench_summary,
        '05_matched_split_incremental_priority_and_context.csv':all_pair,
        '06_paired_rmse_r2_summary.csv':paired_summary,
        '07_matched_partition_manifest_no_ids.csv':pd.DataFrame(member_summary),
    }
    for name,table in outputs.items():
        write_once_csv(results_root/name,table)
    output_hashes={str((results_root/name).relative_to(work)):sha(results_root/name)
                   for name in outputs}
    write_once_json(result_audit,{
        'status':'PASS','signature':signature,'inputs':inputs,'n':6039,
        'repeated_split_seeds':seeds,'blocks_per_split':7,
        'catboost_total_across_splits':70,
        'catboost_reused_from_stage04':7,'new_catboost_fits':63,
        'benchmark_seven_learners':'CatBoost + five other models + mean baseline',
        'noncatboost_trained_estimators_per_seed':5,'new_noncatboost_fits':50,
        'benchmark_rows':70,'block_metric_rows_including_mean_baseline':80,
        'aggregate_output_sha256':output_hashes,
        'private_model_and_responseid_checkpoints':str(model_root),
        'ten_splits_are_not_independent_replications':True,
    })
    print('\n'+'='*78)
    print('PASS | STAGE06 completed: 10 matched splits; original 7 blocks and seven learners')
    print('PASS | 7 existing first-split CatBoost models reused; 63 additional block fits')
    print('FULL SCP 10-SPLIT R2 MEAN:', f'{float(means.loc[means.block=="SCP","mean_r2"].iloc[0]):.9f}')
    print('BENCHMARK RESULTS:',results_root)
    print('AUDIT:',result_audit)
    print('NEXT | Priority null B1000; grouped SHAP and uncertainty; 200-refit response inference')
    print('='*78,flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print('FAILED |',type(error).__name__,str(error),file=sys.stderr,flush=True)
        raise
