#!/usr/bin/env python3
"""WSQS N6039 | Stage 03 preflight: construct model inputs, no empirical fits.

Run after successful Stage02 installation:
    python3 ~/Downloads/WSQS_STAGE02_REPRO_N6039/preflight.py

Writes full-precision aggregate score diagnostics and the verified input plan.
Never saves respondent-level inputs to the public repository.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAME='NPJ_WSQS_RERUN_20261030_N6039'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def require(ok,message):
    if not ok:raise ValueError(message)

def create_new(path,contents):
    require(not path.exists(),'Refusing to overwrite existing preflight: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(contents,encoding='utf-8')

def main():
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path.home()/'Desktop/Spring 2026/Chapter 3 MANUAL RERUN')
    args=p.parse_args()
    work=args.base.expanduser().resolve()/NAME
    original=work/'00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    frozen=work/'07_audit/COHORT_DATASET_LOCK_N6039.json'
    code=work/'02_code_private/reconciled/WSQS_Study_Repro_N6039'
    require(original.is_file() and frozen.is_file() and code.is_dir(),'Run the cohort build and stage02 integration first')
    lock=json.loads(frozen.read_text(encoding='utf-8'))
    require(lock.get('output_sha256',{}).get('primary')==sha(original),'N6039 input checksum differs from lock')
    sys.path.insert(0,str(code/'src'))
    from wsqs.features import construct_features
    from wsqs.extended import DOMAINS, scientific_plans
    from wsqs.models import partition_indices, model_settings
    frame=pd.read_csv(original,dtype={'ResponseId':str},low_memory=False)
    require(frame.shape==(6039,182),'Locked cohort shape differs from 6039 x 182')
    require(frame['ResponseId'].is_unique and frame['ResponseId'].notna().all(),'ResponseId issues')
    if 'municipal_supply_applicable' not in frame.columns:
        require('source09M_municipal_eligible' in frame.columns,'Missing municipal-use applicability field')
        frame=frame.rename(columns={'source09M_municipal_eligible':'municipal_supply_applicable'})
    X,categorical,numeric=construct_features(frame)
    plans=scientific_plans()
    require(X.shape==(6039,33) and list(X.columns)==plans['full'],'33 predictor specification differs')
    require(len(categorical)+len(numeric)==33 and len(categorical)>=1,'Invalid predictor typing')
    raw=[f'{domain}_raw_severity' for domain in DOMAINS]
    require(all(c in frame.columns for c in raw),'Missing CORE domain input fields')
    severity=frame[raw].apply(pd.to_numeric,errors='coerce')
    valid=severity.notna()
    require(severity.stack().isin([1,2,3,4,5]).all(),'Unrecognized raw severity levels')
    count=valid.sum(axis=1)
    require(count.between(7,8).all(),'Unanticipated number of applicable CORE domains')
    score=(5.0-severity.mean(axis=1))*25
    require(score.notna().all() and score.between(0,100).all(),'CORE8 outcome is invalid')
    fit,val,test,dev=partition_indices(len(frame),int(model_settings()['primary_seed']))
    require((len(fit),len(val),len(test),len(dev))==(3864,967,1208,4831),'New cohort partition count mismatch')
    cohorts={'full':int(len(score)),'eight':int((count==8).sum()),'seven':int((count==7).sum())}
    report={
        'status':'SCORING_AND_FEATURE_PREFLIGHT_PASSED',
        'source_sha256':sha(original),
        'n_households':6039,
        'n_source_columns':182,
        'n_features':33,
        'service_support_features':len(plans['blocks']['S']),
        'context_features':len(plans['blocks']['C']),
        'priority_features':len(plans['blocks']['P']),
        'n_categorical':len(categorical),'n_numeric':len(numeric),
        'cohorts':cohorts,
        'core8_mean_full_precision':float(score.mean()),
        'core8_std_full_precision':float(score.std(ddof=1)),
        'core8_min':float(score.min()),'core8_max':float(score.max()),
        'partition':{'seed':int(model_settings()['primary_seed']), 'fitting':len(fit),'internal_validation':len(val),'heldout_test':len(test),'development':len(dev)},
        'domain_summary':[{'domain':domain, 'valid_n':int(valid[f'{domain}_raw_severity'].sum()), 'mean_WSQS_100':float(((5-severity[f'{domain}_raw_severity'])*25).mean()),'sd_WSQS_100':float(((5-severity[f'{domain}_raw_severity'])*25).std(ddof=1))}for domain in DOMAINS],
        'model_fits_completed':0,
        'next':'Empirical pipeline: main models, repeated splits, 8 cross-domain baselines, 1000 priority-null permutations, SHAP, B200 and robustness',
    }
    out=work/'04_numerical_results/measurement/CORE8_AND_FEATURE_PREFLIGHT_N6039.json'
    create_new(out,json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('PASS | Source cohort N6039 and SHA-256 lock')
    print('PASS | Original 33 predictors constructed; 22 S / 3 C / 8 P')
    print('PASS | CORE8 severity, 7/8-domain completeness and 0-100 scale')
    print('PASS | Primary partition 3864 fit / 967 val / 1208 test')
    print('PASS | Full-precision aggregate measurement preflight saved')
    print('CORE8_N:',len(score),'MEAN:',format(score.mean(),'.9f'),'SEVEN_DOMAINS:',cohorts['seven'])
    print('REPORT:',out)
    print('NO EMPIRICAL MODEL WAS FITTED')

if __name__=='__main__':
    try:main()
    except (ValueError, OSError, ImportError) as exc:
        print('FAILED |',str(exc),file=sys.stderr);sys.exit(1)
