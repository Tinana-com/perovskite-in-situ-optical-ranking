"""Frozen CPU study of real D2 four-channel transients. Writes only inside P2."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__file__).resolve().parents[1] / '.mplconfig'))
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
CFG = BASE / 'configs/frozen_p2_protocol.json'
CHANNELS = ['LP725','LP780','SP775','ND']
BASIC = ['mean','last','slope']
ADDED = ['early_trim_mean','late_trim_mean','end_start_delta','late_slope','q10','q90','auc_relative_index','peak_relative_position_smoothed']
PAIRS = [('LP725','LP780'),('LP725','SP775'),('LP780','SP775')]
KEYS = ['sample_id','date','expID','patchID']
SEED = 20261019

def config():
    return json.loads(CFG.read_text(encoding='utf-8'))

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def safe_dir(path):
    p = Path(path).resolve()
    if p == BASE or not p.is_relative_to(BASE):
        raise ValueError('Output must be nested inside P2')
    return p

def validate_source(c):
    p = ROOT / c['data_path']
    if not p.is_file():
        raise FileNotFoundError(f'Missing real D2: {p}. Restore existing data or obtain data.h5 from https://zenodo.org/records/14609789; no simulation fallback.')
    if p.stat().st_size != c['expected_bytes'] or digest(p,'md5') != c['expected_md5']:
        raise ValueError('D2 size/MD5 mismatch')
    return p

def slope(a, x=None):
    if len(a) <= 1:
        return 0.0
    if x is None:
        x = np.linspace(0,1,len(a))
    return float(np.polyfit(x,a,1)[0])

def trace_features(values):
    a = np.asarray(values,dtype=float).reshape(-1)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Signal must be nonempty and finite; do not drop labeled samples silently')
    n = len(a)
    x = np.linspace(0,1,n) if n>1 else np.array([0.0])
    w = max(1,int(np.ceil(n*.25)))
    early = float(trim_mean(a[:w],.1)); late = float(trim_mean(a[-w:],.1))
    smooth = np.convolve(np.pad(a,(2,2),mode='edge'),np.ones(5)/5,mode='valid')
    return dict(zip(BASIC+ADDED,[float(a.mean()),float(a[-1]),slope(a),early,late,late-early,
        slope(a[-w:],x[-w:]),float(np.quantile(a,.1)),float(np.quantile(a,.9)),
        float(np.trapezoid(a,x)) if n>1 else 0.0, float(x[np.argmax(smooth)])]))

def build_features(raw):
    # Strict allowlist: no downstream characterization fields are read here.
    rows=[]
    for channels in raw[CHANNELS].itertuples(index=False,name=None):
        if len({len(a) for a in channels}) != 1:
            raise ValueError('Unequal channel lengths')
        row={}
        for ch,a in zip(CHANNELS,channels):
            row.update({f'{ch}_{k}':v for k,v in trace_features(a).items()})
        rows.append(row)
    f=pd.DataFrame(rows,index=raw.index)
    for col in ['ratio','molarity','evac_duration']:
        f[col]=pd.to_numeric(raw[col],errors='raise')
    return f

def feature_names(family):
    if family=='F0': return []
    if family=='F1': return ['ratio','molarity']
    if family=='F1_evac': return ['ratio','molarity','evac_duration']
    names=[f'{ch}_{k}' for ch in CHANNELS for k in (BASIC if family=='F2' else BASIC+ADDED)]
    if family in ('F4','F4_process'):
        names += [f'ratio_{a}_{b}' for a,b in PAIRS] + [f'ratio_{a}_{b}_missing' for a,b in PAIRS]
    if family=='F4_process': names+=['ratio','molarity']
    if family not in ['F2','F3','F4','F4_process']: raise ValueError(family)
    return names

class FeatureTransform(BaseEstimator,TransformerMixin):
    def __init__(self,family='F2'):
        self.family=family
    def fit(self,X,y=None):
        self.eps_={}
        if self.family in ('F4','F4_process'):
            for a,b in PAIRS:
                v=np.abs(X[f'{b}_late_trim_mean'].to_numpy(float))
                v=v[np.isfinite(v)]
                self.eps_[(a,b)] = max(1e-12,.01*float(np.median(v))) if len(v) else 1e-12
        return self
    def transform(self,X):
        names=feature_names(self.family)
        if self.family not in ('F4','F4_process'):
            return X[names].to_numpy(float)
        out=X.copy()
        for a,b in PAIRS:
            den=X[f'{b}_late_trim_mean'].to_numpy(float)
            num=X[f'{a}_late_trim_mean'].to_numpy(float)
            valid=np.isfinite(num)&np.isfinite(den)&(np.abs(den)>=self.eps_[(a,b)])
            r=np.full(len(X),np.nan)
            np.divide(num,den,out=r,where=valid)
            out[f'ratio_{a}_{b}']=r
            out[f'ratio_{a}_{b}_missing']=(~valid).astype(float)
        return out[names].to_numpy(float)

def make_model(spec,c):
    kind=spec['model']
    args=c['estimators'].get(kind,{}).copy()
    if kind=='ridge': est=Ridge(**args)
    elif kind=='extra_trees': est=ExtraTreesRegressor(random_state=c['seed'],**args)
    elif kind=='hist_gradient_boosting': est=HistGradientBoostingRegressor(random_state=c['seed'],**args)
    else: raise ValueError(kind)
    steps=[('features',FeatureTransform(spec['family'])),('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
    if kind=='ridge': steps.append(('scaler',StandardScaler()))
    return Pipeline(steps+[('model',est)])

def fit_predict(spec,c,x,y,xe):
    if spec['model'] in ('mean','median'):
        fn=np.mean if spec['model']=='mean' else np.median
        return np.full(len(xe),float(fn(y)))
    model=make_model(spec,c)
    model.fit(x,y)
    return model.predict(xe)

def assert_development(dates,c):
    dates=set(map(int,dates))
    if dates & set(c['holdout_dates']) or not dates <= set(c['development_dates']):
        raise ValueError('Holdout or unknown date in development evaluation')

def splits(c):
    dev=set(c['development_dates'])
    flat=sum(c['cv_validation_dates'],[])
    if sorted(flat)!=sorted(dev) or len(flat)!=len(set(flat)):
        raise ValueError('CV dates must cover development dates once')
    for i,val in enumerate(c['cv_validation_dates']):
        assert_development(val,c)
        yield i,sorted(dev-set(val)),val

def prediction_frame(meta,pred,spec,fold,scope,ytrain):
    p=meta[KEYS+['PCE_backward','sequence_length']].copy().rename(columns={'PCE_backward':'y_true'})
    p['y_pred']=pred; p['residual_pp']=pred-p.y_true
    p['abs_error_pp']=np.abs(p.residual_pp); p['squared_error_pp2']=p.residual_pp**2
    p['model_id']=spec['id']; p['feature_family']=spec['family']; p['fold']=fold; p['scope']=scope
    p['seed']=SEED; p['n_train']=len(ytrain); p['risk_threshold_train_q20']=float(np.quantile(ytrain,.2))
    return p

def evaluate(meta,feat,specs,folds,c,scope):
    out=[]
    for fold,trdates,vdates in folds:
        assert_development(trdates+vdates,c)
        if set(trdates)&set(vdates): raise ValueError('Date overlap')
        tr=meta.date.isin(trdates); va=meta.date.isin(vdates)
        if not tr.any() or not va.any(): raise ValueError('Empty grouped split')
        for spec in specs:
            y=meta.loc[tr,'PCE_backward'].to_numpy(float)
            pred=fit_predict(spec,c,feat.loc[tr],y,feat.loc[va])
            out.append(prediction_frame(meta.loc[va],pred,spec,fold,scope,y))
        print(f'{scope} fold {fold}: train={tr.sum()} eval={va.sum()} candidates={len(specs)}',flush=True)
    return pd.concat(out,ignore_index=True)

def metrics(pred):
    group=['scope','model_id','feature_family','seed']
    bydate=pred.groupby(group+['date'],sort=True).agg(n=('y_true','size'),mae_pp=('abs_error_pp','mean'),mse_pp2=('squared_error_pp2','mean')).reset_index()
    bydate['rmse_pp']=np.sqrt(bydate.pop('mse_pp2'))
    byexp=pred.groupby(group+['date','expID'],sort=True).agg(n=('y_true','size'),y_true_mean=('y_true','mean'),y_pred_mean=('y_pred','mean'),patch_mae_pp=('abs_error_pp','mean')).reset_index()
    byexp['mean_pce_abs_error_pp']=abs(byexp.y_true_mean-byexp.y_pred_mean)
    rows=[]
    for keys,p in pred.groupby(group,sort=True):
        sc,mid,fam,seed=keys
        d=bydate[(bydate.scope==sc)&(bydate.model_id==mid)]
        e=byexp[(byexp.scope==sc)&(byexp.model_id==mid)]
        rows.append(dict(zip(group,keys))|{'n':len(p),'n_dates':p.date.nunique(),'n_experiments':len(e),
            'macro_mae_pp':d.mae_pp.mean(),'micro_mae_pp':p.abs_error_pp.mean(),'micro_rmse_pp':np.sqrt(p.squared_error_pp2.mean()),
            'experiment_macro_patch_mae_pp':e.patch_mae_pp.mean(),'experiment_mean_target_mae_pp':e.mean_pce_abs_error_pp.mean(),
            'n_features':len(feature_names(fam))})
    return bydate,byexp,pd.DataFrame(rows).sort_values(['scope','macro_mae_pp','model_id']).reset_index(drop=True)

def write_csv(df,path):
    df.to_csv(path,index=False,float_format='%.12g',lineterminator='\n')

def paired(bydate,pairs,c):
    rows=[]; summaries=[]
    for scope,p in bydate.groupby('scope',sort=True):
        for a,b in pairs:
            xa=p[p.model_id==a].set_index('date').mae_pp
            xb=p[p.model_id==b].set_index('date').mae_pp
            common=sorted(set(xa.index)&set(xb.index))
            if not common: continue
            delta=(xa.loc[common]-xb.loc[common]).to_numpy(float)
            for date,da in zip(common,delta):
                rows.append({'scope':scope,'method_a':a,'method_b':b,'date':date,'delta_mae_pp_a_minus_b':da,'seed':SEED})
            rng=np.random.default_rng(c['seed'])
            boot=rng.choice(delta,size=(c['paired_bootstrap']['replicates'],len(delta)),replace=True).mean(axis=1)
            summaries.append({'scope':scope,'method_a':a,'method_b':b,'n_dates':len(delta),'mean_delta_pp':float(delta.mean()),
                'min_date_delta_pp':float(delta.min()),'max_date_delta_pp':float(delta.max()),'improved_dates':int((delta<0).sum()),
                'worsened_dates':int((delta>0).sum()),'tied_dates':int((delta==0).sum()),
                'date_bootstrap_ci95_low':float(np.quantile(boot,.025)),'date_bootstrap_ci95_high':float(np.quantile(boot,.975)),
                'prespecified': [a,b] in c['paired_prespecified'],'seed':SEED})
    return pd.DataFrame(rows),summaries

def risk_table(pred):
    rows=[]
    for (scope,mid,fold),p in pred.groupby(['scope','model_id','fold'],sort=True):
        cutoff=p.risk_threshold_train_q20.iloc[0]
        truth=p.y_true.to_numpy()<cutoff; flag=p.y_pred.to_numpy()<cutoff
        tp=int((truth&flag).sum()); fn=int((truth&~flag).sum()); fp=int((~truth&flag).sum())
        rows.append({'scope':scope,'model_id':mid,'feature_family':p.feature_family.iloc[0],'fold':fold,'n':len(p),
            'train_q20':cutoff,'tp':tp,'fn':fn,'fp':fp,'low_total':tp+fn,'fnr':fn/(tp+fn) if tp+fn else None,
            'precision':tp/(tp+fp) if tp+fp else None,'seed':SEED,'definition':'regression prediction below train q20; not P1 classifier'})
    return pd.DataFrame(rows)

def p1_replay(meta,feat,c,p1,results):
    """Fixed historical protocol reconciliation only; never a selection input."""
    tr=meta.date.isin(c['development_dates'][:11]); va=meta.date.isin(c['development_dates'][11:]); te=meta.date.isin(c['holdout_dates'])
    old=pd.read_csv(p1/'results/heldout_predictions.csv')
    old=old[old.dataset=='D2']
    vs=pd.read_csv(p1/'results/validation_selection.csv')
    mapping=[('F0_mean','B0_mean'),('F0_median','B0_median'),('F1_ridge','B1_process_no_evac'),('F2_ridge','B2_ridge_1_PL_three_plus_ND')]
    rows=[]
    for new,orig in mapping:
        spec=next(s for s in c['candidates'] if s['id']==new)
        yp=fit_predict(spec,c,feat.loc[tr],meta.loc[tr,'PCE_backward'].to_numpy(float),feat.loc[te])
        vp=fit_predict(spec,c,feat.loc[tr],meta.loc[tr,'PCE_backward'].to_numpy(float),feat.loc[va])
        ref=old[old.model_id==orig].set_index('sample_id').loc[meta.loc[te,'sample_id']]
        if not np.allclose(ref.y_true,meta.loc[te,'PCE_backward'],atol=1e-12): raise ValueError('P1 sample mismatch')
        diff=float(np.max(np.abs(yp-ref.y_pred.to_numpy())))
        mae=float(np.mean(np.abs(yp-meta.loc[te,'PCE_backward'].to_numpy())))
        vmae=float(np.mean(np.abs(vp-meta.loc[va,'PCE_backward'].to_numpy())))
        oldv=float(vs.loc[vs.model_id==orig,'mae_validation'].iloc[0])
        rows.append({'model_id':orig,'p2_equivalent':new,'original_test_mae_pp':float(np.mean(np.abs(ref.y_true-ref.y_pred))),
            'replayed_test_mae_pp':mae,'max_prediction_abs_delta':diff,'replayed_validation_mae_pp':vmae,
            'validation_abs_delta':abs(oldv-vmae),'matched_at_1e_9':diff<1e-9 and abs(oldv-vmae)<1e-9,'n':int(te.sum()),'seed':SEED})
    write_csv(pd.DataFrame(rows),results/'p1_reconciliation.csv')
    return rows

def inventory(c,out):
    p1=ROOT/c['p1_path']
    rows=[]
    data=validate_source(c)
    paths=[data]+[p for p in sorted(p1.rglob('*')) if p.is_file() and not any(x in p.parts for x in ['.venv','__pycache__','.pytest_cache'])]
    for p in paths:
        rows.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p),
            'source_version':c['git_head_at_start']+' plus pre-existing untracked P1 files; SHA256 identifies exact version',
            'role':'D2 real data' if p==data else 'P1 read-only reference','md5':digest(p,'md5') if p==data else '',
            'source_url':'https://zenodo.org/records/14609789' if p==data else 'local P1 result',
            'license':'CC-BY-4.0 per saved Zenodo record' if p==data else 'inherited P1; see source metadata',
            'acquisition_time':'pre-existing; original acquisition time unavailable'})
    write_csv(pd.DataFrame(rows),out/'source_inventory.csv')
    return rows

def figures(cv,dates,held,meta,out,selected):
    out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,5.5)); p=cv.iloc[::-1]
    ax.barh(p.model_id,p.macro_mae_pp,color='#346f91'); ax.set_xlabel('Date equal macro MAE (PCE percentage points)')
    ax.set_title('14-date grouped CV; score used for selection')
    fig.tight_layout(); fig.savefig(out/'fig1_cv_models.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.8))
    p=dates[dates.scope=='GROUPED_CV'].pivot(index='date',columns='model_id',values='mae_pp')
    for mid in ['F2_ridge','F1_ridge','F0_median']:
        ax.plot([str(x) for x in p.index],p[selected]-p[mid],marker='o',label=f'{selected} minus {mid}')
    ax.axhline(0,color='black',lw=1); ax.tick_params(axis='x',rotation=60); ax.set_ylabel('Date MAE difference (pp); negative = improvement')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'fig2_date_deltas.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    p=held[held.model_id==selected]
    for date,part in p.groupby('date'):
        axs[0].scatter(part.y_true,part.y_pred,s=7,alpha=.45,label=str(date))
        axs[1].scatter(part.y_true,part.residual_pp,s=7,alpha=.45)
    axs[0].plot([0,20],[0,20],color='black',lw=1); axs[1].axhline(0,color='black',lw=1)
    axs[0].set(xlabel='Observed PCE (%)',ylabel='Predicted PCE (%)'); axs[1].set(xlabel='Observed PCE (%)',ylabel='Prediction minus observation (pp)')
    axs[0].legend(fontsize=8); fig.suptitle('PREVIOUSLY_INSPECTED_HOLDOUT: frozen selected model')
    fig.tight_layout(); fig.savefig(out/'fig3_residual_failures.png',dpi=160); plt.close(fig)
    table=meta.groupby(['date','sequence_length']).size().unstack(fill_value=0)
    fig,ax=plt.subplots(figsize=(11,5)); table.plot.bar(stacked=True,ax=ax,colormap='tab20')
    ax.set_ylabel('All D2 patches, including missing labels'); ax.legend(title='Sequence points',fontsize=7)
    fig.tight_layout(); fig.savefig(out/'fig4_date_length.png',dpi=160); plt.close(fig)

def run(out):
    out=safe_dir(out)
    if out.exists() and any(out.iterdir()): raise ValueError('Refusing to overwrite an existing run')
    out.mkdir(parents=True,exist_ok=True); results=out/'results'; results.mkdir()
    c=config(); start=time.perf_counter(); cpu=time.process_time()
    sources=inventory(c,results)
    raw=pd.read_hdf(validate_source(c),key='df')
    raw['date']=raw.date.astype(int)
    meta=raw[['date','expID','patchID','PCE_backward','ratio','molarity','evac_duration']].copy()
    meta['sample_id']=[f'D2S_{i+1:05d}' for i in range(len(raw))]
    meta['sequence_length']=raw.ND.map(len).to_numpy()
    if meta.duplicated(['date','expID','patchID']).any(): raise ValueError('Duplicate sample IDs')
    if meta.groupby('expID').date.nunique().max()!=1: raise ValueError('Experiment shared across dates')
    labeled=np.isfinite(meta.PCE_backward.to_numpy(float))
    dates=sorted(meta.loc[labeled,'date'].unique().tolist())
    if dates!=c['development_dates']+c['holdout_dates']: raise ValueError('Frozen labeled dates differ')
    print(f'Data verified: {len(raw)} rows; {labeled.sum()} labels; extracting four-channel features',flush=True)
    feat=build_features(raw)
    write_csv(pd.concat([meta,feat.drop(columns=['ratio','molarity','evac_duration'])],axis=1),results/'extracted_features.csv')
    audit=meta.groupby(['date','ratio','molarity','sequence_length'],dropna=False).agg(n=('sample_id','size'),n_labeled=('PCE_backward','count'),pce_mean=('PCE_backward','mean')).reset_index()
    audit['missing_label_fraction']=1-audit.n_labeled/audit.n
    write_csv(audit,results/'date_length_recipe_audit.csv')
    dictionary=[]
    for family in ['F0','F1','F2','F3','F4','F4_process','F1_evac']:
        for name in feature_names(family):
            dictionary.append({'feature_family':family,'feature':name,'unit':'relative index [0,1]' if 'position' in name else 'signal / relative index' if 'slope' in name else 'AUC relative index; signal unit unverified' if 'auc' in name else 'dimensionless empirical channel ratio' if name.startswith('ratio_') else 'source numeric units unverified',
                'fit_on':'training only' if name.startswith('ratio_') else 'per individual trace; no label',
                'definition':'see frozen protocol and trace_features; ratios have limited empirical interpretation; ND is a sensor channel'})
    write_csv(pd.DataFrame(dictionary),results/'features_dictionary.csv')
    folds=list(splits(c))
    foldrows=[]
    for fold,tr,va in folds:
        for date in c['development_dates']:
            foldrows.append({'fold':fold,'date':date,'role':'validation' if date in va else 'train','seed':SEED,'original_p1_role':'train' if date in c['development_dates'][:11] else 'validation'})
    for date in c['holdout_dates']:
        foldrows.append({'fold':-1,'date':date,'role':c['holdout_status'],'seed':SEED,'original_p1_role':'test'})
    write_csv(pd.DataFrame(foldrows),results/'fold_assignments.csv')
    devmask=labeled&meta.date.isin(c['development_dates'])
    dm=meta.loc[devmask]; dx=feat.loc[devmask]
    with threadpool_limits(limits=1):
        cv=evaluate(dm,dx,c['candidates'],folds,c,'GROUPED_CV')
        bydate,byexp,comparison=metrics(cv)
        selected=comparison.iloc[0].model_id
        enhanced=comparison[comparison.feature_family.isin(['F3','F4','F4_process'])].iloc[0].model_id
        # This immutable selection receipt is written before any P2 holdout fit/evaluation.
        selection={'selected_model':selected,'best_enhanced_model':enhanced,'selected_by':'14-date macro-MAE only',
            'protocol_sha256':digest(CFG),'cv_predictions_sha256_pending':None,'holdout_used_for_selection':False}
        write_csv(cv,results/'cv_predictions.csv')
        selection['cv_predictions_sha256_pending']=digest(results/'cv_predictions.csv')
        dump(results/'selection_freeze.json',selection)
        write_csv(bydate,results/'cv_metrics_by_date.csv'); write_csv(byexp,results/'cv_metrics_by_experiment.csv')
        write_csv(comparison,results/'cv_model_comparison.csv')
        print(f'SELECTION FROZEN: {selected}; best enhanced={enhanced}',flush=True)
        # Predeclared sensitivity analyses cannot change selection.
        ids=set(c['sensitivity']['fixed_length_models']+[selected,enhanced])
        sub_specs=[s for s in c['candidates'] if s['id'] in ids]
        sm=dm[dm.sequence_length==c['sensitivity']['fixed_length']]; sx=dx.loc[sm.index]
        eligible=all(sm.date.isin(va).sum()>0 and sm.date.isin(tr).sum()>0 for _,tr,va in folds)
        sensitivity=[]
        if eligible:
            fix=evaluate(sm,sx,sub_specs,folds,c,'FIXED_LENGTH_719_CV')
            fd,fe,fc=metrics(fix)
            write_csv(fix,results/'fixed_length_predictions.csv'); write_csv(fd,results/'fixed_length_metrics_by_date.csv')
            write_csv(fe,results/'fixed_length_metrics_by_experiment.csv'); write_csv(fc,results/'fixed_length_comparison.csv')
            sensitivity.append(fix)
        forwardfolds=[(i,c['development_dates'][:n],c['development_dates'][n:n+2]) for i,n in enumerate(c['sensitivity']['forward_train_date_counts'])]
        fwids=set(c['sensitivity']['forward_models']+[selected,enhanced])
        fw=evaluate(dm,dx,[s for s in c['candidates'] if s['id'] in fwids],forwardfolds,c,'CHRONOLOGICAL_FORWARD')
        fwd,fwe,fwc=metrics(fw)
        write_csv(fw,results/'forward_predictions.csv'); write_csv(fwd,results/'forward_metrics_by_date.csv')
        write_csv(fwe,results/'forward_metrics_by_experiment.csv'); write_csv(fwc,results/'forward_comparison.csv')
        sensitivity.append(fw)
        # Duration is a leakage-sensitive separate comparison, never a main candidate.
        evac={'id':'F1_evac_ridge_SENSITIVITY_ONLY','family':'F1_evac','model':'ridge'}
        ev=evaluate(dm,dx,[evac],folds,c,'EVAC_DURATION_SENSITIVITY')
        evd,eve,evc=metrics(ev); write_csv(evc,results/'evac_duration_comparison.csv'); write_csv(evd,results/'evac_duration_by_date.csv')
        holdids=set(c['holdout_controls']+[selected,enhanced])
        te=labeled&meta.date.isin(c['holdout_dates'])
        hrows=[]
        for spec in c['candidates']:
            if spec['id'] in holdids:
                y=dm.PCE_backward.to_numpy(float)
                hp=fit_predict(spec,c,dx,y,feat.loc[te])
                hrows.append(prediction_frame(meta.loc[te],hp,spec,-1,c['holdout_status'],y))
        held=pd.concat(hrows,ignore_index=True)
        hd,he,hc=metrics(held)
        write_csv(held,results/'frozen_model_heldout_predictions.csv'); write_csv(hc,results/'heldout_metrics.csv')
        write_csv(hd,results/'heldout_metrics_by_date.csv'); write_csv(he,results/'heldout_metrics_by_experiment.csv')
        reconciliation=p1_replay(meta.loc[labeled],feat.loc[labeled],c,ROOT/c['p1_path'],results)
    pairs=c['paired_prespecified']+[[selected,b] for b in ['F0_mean','F0_median','F1_ridge','F2_ridge'] if b!=selected]
    alldates=pd.concat([bydate,hd]+[metrics(p)[0] for p in sensitivity],ignore_index=True)
    deltas,ps=paired(alldates,pairs,c)
    write_csv(deltas,results/'paired_date_deltas.csv'); write_csv(pd.DataFrame(ps),results/'paired_delta_summary.csv')
    write_csv(risk_table(pd.concat([cv,held],ignore_index=True)),results/'secondary_risk_metrics.csv')
    # Same-model feature ablations; all main candidates retained, including negative results.
    abl=[]
    for model in ['ridge','extra_trees','hist_gradient_boosting']:
        b=comparison[comparison.model_id==f'F2_{model}'].iloc[0]
        for family in ['F3','F4','F4_process']:
            p=comparison[comparison.model_id==f'{family}_{model}']
            if len(p):
                a=p.iloc[0]
                abl.append({'scope':'GROUPED_CV','model_id':a.model_id,'feature_family':family,'baseline_id':b.model_id,
                    'n':int(a.n),'n_dates':int(a.n_dates),'macro_mae_pp':a.macro_mae_pp,'delta_macro_mae_pp':a.macro_mae_pp-b.macro_mae_pp,'seed':SEED})
    write_csv(pd.DataFrame(abl),results/'ablation.csv')
    cases=held[held.model_id==selected].sort_values(['abs_error_pp','sample_id'],ascending=[False,True]).head(30)
    cases=cases.merge(meta[KEYS+['ratio','molarity']],on=KEYS,validate='one_to_one')
    write_csv(cases,results/'failure_cases.csv')
    figures(comparison,bydate,held,meta,out/'figures',selected)
    # Small REAL trace example; not synthetic data and not model training subsample.
    sample=raw.loc[dm.index[0]]
    write_csv(pd.DataFrame({'relative_index':np.linspace(0,1,len(sample.ND)),**{ch:sample[ch] for ch in CHANNELS}}),results/'small_real_trace_example.csv')
    summary={'status':'ENGINEERING_COMPLETE','seed':SEED,'protocol_sha256':digest(CFG),
        'git_head':c['git_head_at_start'],'D2_path':str(ROOT/c['data_path']),'D2_sha256':sources[0]['sha256'],
        'rows_total':len(meta),'rows_labeled':int(labeled.sum()),'dates_total':int(meta.date.nunique()),'dates_labeled':len(dates),
        'experiments_total':int(meta.groupby(['date','expID']).ngroups),'duplicate_keys':int(meta.duplicated(['date','expID','patchID']).sum()),
        'missing_labels':int((~labeled).sum()),'entirely_unlabeled_dates':[int(d) for d,p in meta.groupby('date') if p.PCE_backward.notna().sum()==0],
        'development_rows':len(dm),'holdout_rows':int(te.sum()),'development_dates':c['development_dates'],'holdout_dates':c['holdout_dates'],
        'selected_model':selected,'best_enhanced_model':enhanced,'cv_comparison':comparison.to_dict('records'),
        'holdout_status':c['holdout_status'],'heldout_comparison':hc.to_dict('records'),
        'paired_comparisons':ps,'fixed_length_719_eligible':eligible,'fixed_length_719_rows':len(sm),'fixed_length_719_dates':int(sm.date.nunique()),
        'p1_reconciliation':reconciliation,'main_candidates':len(c['candidates']),'absolute_time_verified':False,'full_spectrum':False,
        'sensitivity_used_for_reselection':False,'prediction_clipping':False,
        'wall_seconds':round(time.perf_counter()-start,3),'cpu_seconds':round(time.process_time()-cpu,3)}
    dump(results/'numeric_summary.json',summary)
    dump(out/'run_receipt.json',{'command':subprocess.list2cmdline(sys.argv),'python':sys.version,'executable':sys.executable,
        'platform':platform.platform(),'protocol_sha256':digest(CFG),'wall_seconds':summary['wall_seconds'],'cpu_seconds':summary['cpu_seconds'],
        'status':'SUCCESS','all_model_hyperparameters':'configs/frozen_p2_protocol.json','threads':1,'raw_data_downloads':0})
    print(json.dumps({'status':'SUCCESS','selected':selected,'wall_seconds':summary['wall_seconds']},ensure_ascii=False),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output)

if __name__=='__main__': main()
