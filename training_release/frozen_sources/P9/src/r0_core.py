"""P9 R0: target-free sequence utilities and development-only M0 adapter."""
import os,sys,json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parents[1]
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']: os.environ[n]='1'
os.environ['MPLCONFIGDIR']=str(BASE/'.mplconfig'); os.environ['NUMBA_CACHE_DIR']=str(BASE/'.numba_cache')
sys.path.insert(0,str(ROOT/'research/pl_public_data_pilot_06_baselines_ablations/.deps'))
sys.path.insert(0,str(BASE/'.deps'))
import numpy as np
import pandas as pd
import p6_core as p
KEYS=p.KEYS; METRICS=p.METRICS; CHANNELS=['LP725','LP780','SP775','ND']; PL_CHANNELS=CHANNELS[:3]
sha=p.sha; dump=p.dump
def csv(x,path): x.to_csv(path,index=False,float_format='%.17g',lineterminator='\n')
def config(): return json.loads((BASE/'configs/frozen_p9_protocol.json').read_text())
def safe(path):
    q=Path(path).resolve()
    if not q.is_relative_to(BASE) or q==BASE: raise ValueError('P9 output boundary')
    return q
def resample_shape(values):
    """No labels or fitted/global statistics. Relative-index linear interpolation."""
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1: raise ValueError('Only actual one-dimensional channel traces')
    if len(a)==0: raise ValueError('EMPTY_TRACE blocks that branch; never remove row')
    finite=np.isfinite(a); status={'n_nonfinite':int((~finite).sum()),'all_nonfinite':not bool(finite.any()),'constant':False}
    if finite.any(): a=np.interp(np.arange(len(a)),np.flatnonzero(finite),a[finite])
    else: a=np.zeros(len(a))
    b=np.interp(np.linspace(0,1,256),np.linspace(0,1,len(a)),a)
    lo=b.min(); span=b.max()-lo
    if span<=1e-12: z=np.zeros(256); status['constant']=True
    else: z=(b-lo)/span
    return np.ascontiguousarray(z,dtype=np.float32),status

def load_development(audit_raw=False):
    c=p.config(); sources=json.loads((BASE/'RUN_RECEIPTS/input_hash_verification.json').read_text())
    for v in sources:
        q=Path(v['path']); assert q.stat().st_size==v['bytes'] and sha(q)==v['sha256']
    d1=pd.read_hdf(ROOT/c['D1']['data_directory']/c['D1']['train_file']['name'],'df')
    ids=p.d1_identity(d1); opt=ids.copy(); opt['LP725']=d1.LP725.to_numpy(); labels=ids.copy(); labels['target']=d1.pce_mean.to_numpy(float)
    folds=ids.copy(); folds['outer_fold']=d1.foldNo.to_numpy(int)
    assert len(d1)==780 and labels.unit_id.nunique()==26 and d1.groupby('substrateID').foldNo.nunique().max()==1
    hist=pd.read_csv(ROOT/'research/pl_public_data_pilot_06_baselines_ablations/results/d1_author_test_predictions.csv',usecols=KEYS)
    assert len(hist)==349 and hist.unit_id.nunique()==12 and not set(hist.unit_id)&set(labels.unit_id)
    d={'D1':{'opt':opt,'y':labels,'folds':folds,'features':p.d1_features(opt),'raw_opt':opt,'schema':list(d1.columns),'history_identity':hist}}
    # HDF fixed-format container is decoded once; historical target columns are never projected/scored.
    raw=pd.read_hdf(ROOT/c['D2']['data_path'],'df'); cache=pd.read_csv(ROOT/c['D2']['features_path'])
    assert len(raw)==len(cache)==11136
    for n in ['date','expID','patchID']: assert np.array_equal(raw[n].to_numpy(),cache[n].to_numpy())
    assert not cache.sample_id.duplicated().any() and not raw.duplicated(['date','expID','patchID']).any()
    meta=pd.DataFrame({'sample_id':cache.sample_id,'unit_id':cache.date.astype(str)+'::'+cache.expID,'split_group':cache.date.astype(str),'patchID':cache.patchID.astype(int)})
    dev=cache.date.isin(c['D2']['development_dates']); history=cache.date.isin(c['D2']['history_dates'])
    y=meta.loc[dev].copy(); y['target']=raw.loc[dev,'PCE_backward'].to_numpy(float)
    assert np.allclose(y.target,cache.loc[dev,'PCE_backward'],equal_nan=True,atol=0,rtol=0)
    counts=y.groupby('unit_id').target.count(); y=y[y.target.notna()&y.unit_id.isin(counts[counts>=4].index)].reset_index(drop=True)
    opt=pd.concat([meta.loc[dev].reset_index(drop=True),raw.loc[dev,CHANNELS].reset_index(drop=True)],axis=1)
    f=pd.concat([meta.loc[dev].reset_index(drop=True),cache.loc[dev,p.p4.F2].reset_index(drop=True)],axis=1)
    # Recompute original F2 from actual raw arrays, preserving original polyfit semantics.
    vals={}
    for ch in CHANNELS:
        traces=raw.loc[dev,ch]
        vals[ch+'_mean']=traces.map(lambda a:float(np.asarray(a,float).mean())).to_numpy()
        vals[ch+'_last']=traces.map(lambda a:float(a[-1])).to_numpy()
        vals[ch+'_slope']=traces.map(lambda a:float(np.polyfit(np.linspace(0,1,len(a)),np.asarray(a,float),1)[0])).to_numpy()
    recomputed=pd.DataFrame(vals); diffs={n:float(abs(recomputed[n].to_numpy()-f[n].to_numpy()).max()) for n in p.p4.F2}
    assert max(diffs.values())<1e-7,'Raw D2 F2 cache mismatch; block, do not refit to new features'
    folds=y[KEYS].copy(); mapping={str(v):i for i,vs in enumerate(c['D2']['outer_validation_dates']) for v in vs}; folds['outer_fold']=folds.split_group.map(mapping)
    assert (len(y),y.unit_id.nunique(),y.split_group.nunique())==(4707,158,14)
    assert np.isfinite(y.target).all() and (y.target==0).sum()==33
    rawopt=pd.concat([meta,raw[CHANNELS]],axis=1)
    d['D2']={'opt':opt,'y':y,'features':f,'folds':folds,'raw_opt':rawopt,'schema':list(raw.columns),'cache_max_differences':diffs,'history_identity':meta.loc[history].reset_index(drop=True)}
    assert all(not set(q['y'].split_group)&set(q['history_identity'].split_group) for q in d.values())
    return d

def fit_predict_m0(dataset,features,labels,validation_groups):
    """Projects labels to training groups BEFORE model fitting; optical API is strict."""
    val=set(map(str,validation_groups)); groups=set(features.split_group); training=groups-val
    y=labels[labels.split_group.isin(training)].copy(); tr=features[features.split_group.isin(training)].copy(); ev=features[features.split_group.isin(val)].copy()
    if not val or val&set(y.split_group) or set(tr.unit_id)&set(ev.unit_id): raise ValueError('Group separation')
    spec=next(x for x in p.grid() if x['dataset']==dataset and x['model_id']=='PAIR_ABS')
    model=p.Model(dataset,spec,p.config()).fit(y,tr); scores=ev[KEYS].copy(); scores['score']=model.predict(ev)
    return scores,model

def independent(metrics,dataset): return p.resampling_units(metrics,dataset)
