#!/usr/bin/env python3
"""Stage13 | Privacy-conscious missing aggregate exhibit sources, NO CatBoost fitting.

Computes ONLY the plotting/table cells that Stage12's 47-file aggregate handoff lacks,
using the immutable N6039 source and existing locked private Stage08/Stage10 outputs.
Keeps ResponseIds, individual WSQS and individual SHAP values PRIVATE on author's Mac.

This supplies F2d, F3a/b/c, F4a 750-within-group bootstrap, Table2 joint-priority
heterogeneity, Table3 95th percentile absolute shifts, and Table5 net absolute block
SHAP. Exact Figure5e and SI Figure S4 scatter points stay private and must be
rendered locally from their original locked private sources when figures are assembled.

New bootstrap seed documented; historical private figure-specific RNG unverified.
No historical/model output is ever altered.
"""
from __future__ import annotations
import hashlib,json,sys,os,tempfile
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT='NPJ_WSQS_RERUN_20261030_N6039'
STAGE='STAGE13_PLOT_SOURCE_GAPS_N6039'
EXPECTED_SHA='c72b8480d3cff223d1d4cf0ebcc29a3dd80204d563cd63a4295f6739f0ed77a1'
B=750
SEED=20260714
DOMAINS=['Affordability','Access','Reliability','Efficiency','Acceptability','Quality','Quantity','Customer_service']

def require(x,s):
    if not x: raise RuntimeError(s)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)

def write(p,obj):
    p=Path(p)
    require(not p.exists(),'Refusing to overwrite '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=p.parent,suffix='.part',delete=False) as f:
        tmp=Path(f.name)
        if isinstance(obj,pd.DataFrame):obj.to_csv(f,index=False,float_format='%.17g')
        else:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
    os.replace(tmp,p)

def calculate(work):
    work=Path(work).expanduser().resolve()
    data=work/'00_private_inputs/cohort/WSQS_PRIMARY_ANALYTICAL_N6039.csv'
    stage08audit=work/'07_audit/STAGE08_SHAP_ATTRIBUTION_N6039_COMPLETE.json'
    stage10audit=work/'07_audit/STAGE10_MEASUREMENT_PROVIDER_N6039_COMPLETE.json'
    stage08dir=work/'04_numerical_results/inference/STAGE08_SHAP_ATTRIBUTION_N6039'
    stage10dir=work/'04_numerical_results/measurement/STAGE10_MEASUREMENT_PROVIDER_N6039'
    private08=work/'03_run/checkpoints/STAGE08_SHAP_ATTRIBUTION_N6039/heldout_shap_and_predictors_PRIVATE.csv'
    private10=stage10dir/'16_weighting_household_scores_PRIVATE.csv'
    outdir=work/'04_numerical_results/figure_sources'/STAGE
    audit=work/'07_audit'/(STAGE+'_COMPLETE.json')
    for p in [data,stage08audit,stage10audit,private08,private10,stage08dir/'02_all_33_predictors_shap_with_bootstrap_ci.csv',stage08dir/'01_grouped_shap_with_bootstrap_ci.csv',stage10dir/'17_primary_provision_domain_profiles.csv']:
        require(p.is_file(),'Missing prerequisite '+str(p))
    require(sha(data)==EXPECTED_SHA,'Frozen dataset SHA mismatch')
    a8,a10=readj(stage08audit),readj(stage10audit)
    require(a8['status']==a10['status']=='PASS','Stage08/10 audit not complete')
    require(sha(private10)==a10['output_sha256'][private10.name],'Stage10 private source checksum mismatch')
    require(sha(private08)==a8['checkpoint_sha256'][private08.name],'Stage08 private source checksum mismatch')
    require(sha(stage08dir/'01_grouped_shap_with_bootstrap_ci.csv')==a8['output_sha256']['01_grouped_shap_with_bootstrap_ci.csv'],'Stage08 grouped SHAP changed')
    source_hashes={'cohort':sha(data),'stage08_audit':sha(stage08audit),'stage10_audit':sha(stage10audit),'private08':sha(private08),'private10':sha(private10),'script':sha(Path(__file__).resolve()),'B':B,'seed':SEED}
    sig=hashlib.sha256(canonical(source_hashes).encode()).hexdigest()
    if audit.exists():
        a=readj(audit)
        require(a['status']=='PASS' and a['signature']==sig,'Existing Stage13 audit differs from current code or sources')
        for n,s in a['output_sha256'].items():require((outdir/n).is_file() and sha(outdir/n)==s,'Stage13 output checksum mismatch: '+n)
        print('PASS | Stage13 already complete; all hashes verified, no recomputation')
        return
    require(not outdir.exists() or not any(outdir.iterdir()),'Stage13 output folder not empty without valid audit')

    d=pd.read_csv(data,low_memory=False,dtype={'ResponseId':str})
    w=pd.read_csv(private10,dtype={'ResponseId':str})
    s=pd.read_csv(private08,dtype={'ResponseId':str})
    require(d.shape==(6039,182),'Cohort shape mismatch')
    require(len(w)==6039 and len(s)==1208,'Private source row count mismatch')
    require(d.ResponseId.is_unique and np.array_equal(d.ResponseId.to_numpy(),w.ResponseId.to_numpy()),'Private weighting row order differs')
    require(s.ResponseId.is_unique and s.ResponseId.isin(d.ResponseId).all(),'Private SHAP row membership mismatch')
    sev=pd.DataFrame({k:pd.to_numeric(d[k+'_raw_severity'],errors='coerce') for k in DOMAINS})
    score=(5-sev)*25
    equal=score.mean(axis=1)
    require(np.allclose(equal.to_numpy(),w.CORE8.to_numpy(float),atol=1e-10,rtol=0),'Stage10 private WSQS scoring differs')
    priority=pd.DataFrame({k:pd.to_numeric(d[k+'_priority_0_10'],errors='raise') for k in DOMAINS})
    require(priority.notna().all().all() and priority.stack().between(0,10).all(),'Invalid priority rating')
    vary=priority.nunique(axis=1).gt(1)
    summary={'N':6039,'n_priority_differentiated':int(vary.sum()),'pct_priority_differentiated':float(100*vary.mean()),'n_same_priority_all_domains':int((~vary).sum()),'pct_same_priority_all_domains':float(100*(~vary).mean()),'definition':'At least two of eight priority ratings differ within a household'}

    diff=(w.HOUSEHOLD_PRIORITY-w.CORE8).to_numpy(float)
    edges=np.arange(-60,61.01,1.0)
    ct,_=np.histogram(diff,bins=edges)
    hist=pd.DataFrame({'bin_start_inclusive':edges[:-1],'bin_end_exclusive':edges[1:],'households':ct})
    require(ct.sum()==6039,'F2 weighted-difference histogram incomplete')

    R=score.corr(method='pearson')
    pairs=pd.DataFrame([{'domain_row':a,'domain_col':b,'pearson_r':float(R.loc[a,b]),'pairwise_complete_n':int(score[a].notna().mul(score[b].notna()).sum())} for a in DOMAINS for b in DOMAINS])
    require(len(pairs)==64 and np.allclose(R,R.T),'F3 Pearson matrix invalid')
    qbin=pd.qcut(equal,q=5,labels=False,duplicates='drop')
    require(qbin.notna().all() and len(qbin.unique())==5,'Exactly five distinct WSQS quintiles required')
    qrows=[]
    for g in range(5):
        m=qbin.eq(g)
        for k in DOMAINS:
            x=score.loc[m,k]
            qrows.append({'quintile_low_to_high':g+1,'n_households':int(m.sum()),'domain':k,'n_valid_domain':int(x.notna().sum()),'mean_domain_score':float(x.mean()),'WSQS_quintile_min':float(equal[m].min()),'WSQS_quintile_max':float(equal[m].max())})
    quin=pd.DataFrame(qrows)
    ecdf=[]
    cutoffs=np.linspace(0,100,401)
    for name in ['CORE8','POP_WEIGHT_FULL_SAMPLE','HOUSEHOLD_PRIORITY']:
        a=np.sort(w[name].to_numpy(float))
        for cutoff,n_under in zip(cutoffs,np.searchsorted(a,cutoffs,side='right')):
            ecdf.append({'scheme':name,'WSQS_100':float(cutoff),'cumulative_fraction_le':float(n_under/len(a))})
    ecdf=pd.DataFrame(ecdf)

    labels=d['Q 1.1_raw_value'].astype('string').str.strip()
    saved=pd.read_csv(stage10dir/'17_primary_provision_domain_profiles.csv')
    boots=[]
    for j,row in saved.iterrows():
        label=row['primary_domestic_provision']
        arr=equal.loc[labels.eq(label)].to_numpy(float)
        require(len(arr)==int(row['n']) and np.isclose(arr.mean(),float(row['CORE8_mean']),atol=1e-10),'Provision profile means/n differ '+label)
        rng=np.random.default_rng(SEED+300+j)
        m=np.empty(B,dtype=float)
        for b in range(B):m[b]=arr[rng.integers(0,len(arr),len(arr))].mean()
        lo,hi=np.quantile(m,[.025,.975])
        boots.append({'primary_domestic_provision':label,'n':len(arr),'mean_score_100':float(arr.mean()),'bootstrap_ci2p5':float(lo),'bootstrap_ci97p5':float(hi),'B':B,'seed':SEED+300+j,'bootstrap_definition':'750 within-provision household resamples with replacement'})
    boots=pd.DataFrame(boots)

    shifts=[]
    for name in ['CORE8','POP_WEIGHT_FULL_SAMPLE','HOUSEHOLD_PRIORITY']:
        abs_delta=(w[name]-w.CORE8).abs().to_numpy(float)
        shifts.append({'variant':name,'n':len(abs_delta),'p95_abs_shift_100':float(np.quantile(abs_delta,.95)),'mean_abs_shift_100':float(abs_delta.mean()),'mean_signed_shift_100':float((w[name]-w.CORE8).mean()),'pearson_with_CORE8':float(w[name].corr(w.CORE8))})
    shifts=pd.DataFrame(shifts)

    shap_feats=pd.read_csv(stage08dir/'02_all_33_predictors_shap_with_bootstrap_ci.csv')
    blocks=[]
    for block in ['S','C','P']:
        members=shap_feats.loc[shap_feats.block.eq(block),'feature'].tolist()
        require(len(members)=={'S':22,'C':3,'P':8}[block],'Unexpected SHAP block feature count')
        cols=['shap_'+x for x in members]
        require(all(c in s.columns for c in cols),'SHAP feature missing for '+block)
        mat=s[cols].to_numpy(float)
        net=float(np.abs(mat.sum(axis=1)).mean())
        gross=float(np.abs(mat).mean(axis=0).sum())
        from_canned=float(shap_feats.loc[shap_feats.block.eq(block),'mean_abs_shap_wsqs_points'].sum())
        require(np.isclose(gross,from_canned,atol=1e-9),'Grouped gross SHAP discrepancy '+block)
        blocks.append({'block':block,'feature_count':len(members),'n_heldout':len(s),'gross_absolute_shap_100':gross,'net_absolute_block_shap_100':net})
    nets=pd.DataFrame(blocks)

    outs={
       '01_F2_priority_rating_heterogeneity.json':summary,
       '02_F2d_individual_minus_equal_histogram.csv':hist,
       '03_F3a_domain_pearson_correlations.csv':pairs,
       '04_F3b_equal_WSQS_quintile_domain_profiles.csv':quin,
       '05_F3c_weighting_ecdf_grid.csv':ecdf,
       '06_F4a_provision_mean_bootstrap_B750.csv':boots,
       '07_TABLE03_p95_absolute_shift_and_correlations.csv':shifts,
       '08_TABLE05_net_absolute_block_SHAP.csv':nets,
    }
    outdir.mkdir(parents=True,exist_ok=True)
    for name,table in outs.items():write(outdir/name,table)
    hashes={name:sha(outdir/name) for name in outs}
    payload={'status':'PASS','signature':sig,'signature_inputs':source_hashes,'outputs':list(outs),'output_sha256':hashes,'models_fitted':0,'heldout_rows_exported':0,'respondent_ids_exported':0,'B_provision_bootstrap':B,'note':'New within-group B750 RNG streams explicit; original historical private RNG streams not verified. Private scatter and SHAP feature-level rows NOT exported.'}
    write(audit,payload)
    print('='*74)
    print('PASS | STAGE13 missing aggregate plot-source and table-column recovery')
    print('SOURCE SHA256:',EXPECTED_SHA)
    print('PRIORITY DIFFERENTIATED:',summary['n_priority_differentiated'],'/',6039,'('+format(summary['pct_priority_differentiated'],'.2f')+'%)')
    print('NET ABS SHAP: '+', '.join(f"{r.block}={r.net_absolute_block_shap_100:.6f}" for r in nets.itertuples()))
    print('PASS | F3 correlations/quintiles/ECDF, F2 histogram, F4 B750, T3 p95 shifts')
    print('PASS | No fitting, respondent rows or household-level SHAP exported')
    print('RESULTS:',outdir)
    print('AUDIT:',audit)
    print('='*74)

if __name__=='__main__':
    base=Path.home()/'Desktop/Spring 2026/Chapter 3 MANUAL RERUN'/PROJECT
    calculate(base)
