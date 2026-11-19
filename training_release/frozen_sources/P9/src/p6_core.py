"""Frozen small-model families and optical-only feature/normalization APIs."""
import os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE/'.deps'))
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
os.environ['MPLCONFIGDIR']=str(BASE/'.mplconfig')
import hashlib,json,time,warnings,ctypes
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from lightgbm import LGBMRanker
import p5_reference as p5
import p4_reference as p4
from ranking_metrics import group_metrics,rho,top_k,order

SEED=20261019; KEYS=['sample_id','unit_id','split_group','patchID']; METRICS=['top25_gain_pp','rho','top25_overlap']
def config(): return json.loads((BASE/'configs/frozen_p6_protocol.json').read_text(encoding='utf-8'))
def grid(): return pd.read_csv(BASE/'configs/model_grid.csv').to_dict('records')
def sha(path): return p5.sha(path)
def dump(path,value): p5.dump(path,value)
def csv(df,path): p5.csv(df,path)
def safe_dir(path):
    path=Path(path).resolve()
    if path==BASE or not path.is_relative_to(BASE): raise ValueError('P6 write boundary')
    return path
def require(frame,cols):
    if set(frame.columns)!=set(cols): raise ValueError('Optical prediction API rejects targets and forbidden columns')
    if frame.sample_id.duplicated().any(): raise ValueError('Duplicate sample ID')
    if frame.duplicated(['unit_id','patchID']).any(): raise ValueError('Duplicate unit+patch')
    if frame.groupby('unit_id').split_group.nunique().max()>1: raise ValueError('Unit spans split groups')

def d1_identity(raw):
    q=p5.identity(raw); q['unit_id']=q.substrateID; q['split_group']=q.substrateID
    return q[KEYS]
def clean_window(sequence,window):
    a=np.asarray(sequence,dtype=float)
    if a.shape!=(719,): raise ValueError('D1 full source must contain719 frames')
    w=int(np.ceil(.25*len(a)))
    if window=='EARLY25': a=a[:w]
    elif window=='LATE25': a=a[-w:]
    elif window!='FULL': raise ValueError('Unknown window')
    keep=np.isfinite(a)
    if keep.all(): return a.copy()
    if not keep.any(): return np.zeros(len(a))
    idx=np.arange(len(a)); return np.interp(idx,idx[keep],a[keep])
def d1_features(opt,window='FULL'):
    require(opt,KEYS+['LP725']); a=np.vstack([clean_window(x,window) for x in opt.LP725]); n=a.shape[1]; w=int(np.ceil(.1*n)); early=a[:,:w].mean(1); late=a[:,-w:].mean(1)
    t=np.linspace(0,1,n); centered=t-t.mean()
    vals=np.column_stack([a.mean(1),a.std(1),early,late,late-early,a.max(1),a.argmax(1)/(n-1),(late-early)/np.maximum(abs(early),1e-12),a[:,int(np.floor(.25*n)):int(np.ceil(.75*n))].mean(1),a@centered/(centered@centered),np.trapezoid(a,dx=1,axis=1),abs(np.diff(a,axis=1)).sum(1)])
    if not np.isfinite(vals).all(): raise ValueError('Nonfinite optical statistic')
    return pd.concat([opt[KEYS].reset_index(drop=True),pd.DataFrame(vals,columns=p5.STATS)],axis=1)

def training_weights(y):
    require(y,KEYS+['target'])
    n_unit=y.groupby('unit_id').unit_id.transform('size').to_numpy(float)
    units=y[['split_group','unit_id']].drop_duplicates().groupby('split_group').size()
    w=1/(n_unit*y.split_group.map(units).to_numpy(float)*y.split_group.nunique())
    return w/w.mean()

class Preprocessor:
    def __init__(self,dataset,spec,c): self.ds=dataset; self.spec=spec; self.c=c
    def fit(self,f):
        stats=p5.STATS if self.ds=='D1' else p4.F2; require(f,KEYS+stats)
        self.stats_=[] if self.spec['position']=='ID' else (stats if self.spec['statistics_subset']=='ALL' else self.c['level_statistics' if self.spec['statistics_subset']=='LEVEL' else 'dynamic_statistics'])
        self.floor_=np.maximum(1e-12,.01*np.median(abs(f[self.stats_].to_numpy()),axis=0)) if self.stats_ else np.array([])
        self.names_=[]
        if self.spec['position'] in ['ID','IDPL']: self.names_ += [f'ID_{v}' for v in self.c['categories']]
        if self.spec['position']!='ID':
            for kind in (['ABS','REL'] if self.spec['representation']=='BOTH' else [self.spec['representation']]): self.names_ += [kind+'_'+s for s in self.stats_]
        self.scaler_=StandardScaler().fit(self.block(f)); return self
    def block(self,f):
        require(f,KEYS if self.spec['position']=='ID' and set(f.columns)==set(KEYS) else KEYS+(p5.STATS if self.ds=='D1' else p4.F2)); blocks=[]
        if self.spec['position'] in ['ID','IDPL']: blocks.append((f.patchID.to_numpy()[:,None]==np.array(self.c['categories'])[None,:]).astype(float))
        if self.spec['position']!='ID':
            x=f[self.stats_]
            if self.spec['representation'] in ['ABS','BOTH']: blocks.append(x.to_numpy(float))
            if self.spec['representation'] in ['REL','BOTH']:
                med=f.groupby('unit_id',sort=False)[self.stats_].transform('median'); mad=(x-med).abs().groupby(f.unit_id,sort=False).transform('median')
                blocks.append((x.to_numpy()-med.to_numpy())/np.maximum(1.4826*mad.to_numpy(),self.floor_))
        result=np.column_stack(blocks)
        if not np.isfinite(result).all(): raise ValueError('Nonfinite feature block')
        return result
    def transform(self,f): return self.scaler_.transform(self.block(f))

def pairs(y,x,ds):
    if ds=='D1':
        a=y[KEYS].copy(); a['substrateID']=a.unit_id; a['pce_mean']=y.target.to_numpy(); a=a[p5.KEYS+['pce_mean']]
        return p5.construct_pairs(a,x,.5)
    a=y[KEYS].copy(); a['date']=a.split_group.astype(int); a['expID']=a.unit_id; a['PCE_backward']=y.target.to_numpy()
    return p4.construct_pairs(a[p4.KEYS+['PCE_backward']],x,.5)

def rank_queries(y,x):
    """Supervised relevance constructed from TRAIN labels only; stable identity order."""
    require(y,KEYS+['target']); seq=np.lexsort((y.sample_id.to_numpy(str),y.unit_id.to_numpy(str)))
    q=y.iloc[seq].reset_index(drop=True); ranks=q.groupby('unit_id').target.rank(method='average'); count=q.groupby('unit_id').target.transform('size')
    rel=np.clip(np.floor(5*(ranks-1)/np.maximum(count-1,1)),0,4).to_numpy(np.int32)
    sizes=q.groupby('unit_id',sort=False).size().to_numpy(int)
    if sizes.sum()!=len(y) or len(sizes)!=y.unit_id.nunique(): raise ValueError('Invalid query lengths')
    return x[seq],rel,sizes,seq,q

def peak_memory_mib():
    if os.name!='nt': return None
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(s,ctypes.c_size_t) for s in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    counter=PMC(); counter.cb=ctypes.sizeof(counter); kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=wintypes.HANDLE
    api=ctypes.WinDLL('psapi').GetProcessMemoryInfo; api.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
    return float(counter.PeakWorkingSetSize/(1024**2)) if api(kernel.GetCurrentProcess(),ctypes.byref(counter),counter.cb) else None

class Model:
    def __init__(self,ds,spec,c): self.ds=ds; self.spec=spec; self.c=c
    def fit(self,y,f):
        if not set(y.sample_id)<=set(f.sample_id) or not set(y.split_group)<=set(f.split_group): raise ValueError('Training identity mismatch')
        self.pre_=Preprocessor(self.ds,self.spec,self.c).fit(f)
        lookup=pd.DataFrame(self.pre_.transform(f),index=f.sample_id); yy=y.sort_values(['split_group','unit_id','sample_id']).reset_index(drop=True)
        # D1 lexicographic groups coincide with original P5 ordering; D2 original date order.
        x=lookup.loc[yy.sample_id].to_numpy(); w=training_weights(yy); fam=self.spec['family']; self.pair_ids_=[]; self.pair_audit_=None; self.query_audit_=None
        if fam=='PAIRWISE':
            xx,labels,ww,audit,ids=pairs(yy,x,self.ds); self.pair_ids_=ids; self.pair_audit_=audit
            self.pair_hash_=hashlib.sha256('\n'.join(ids).encode()).hexdigest(); self.pair_weight_sum_=float(ww.sum()); self.n_pairs_=len(labels)
            self.model_=LogisticRegression(random_state=SEED,**self.c['models']['PAIRWISE'])
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning); self.model_.fit(xx,labels,sample_weight=ww)
        elif fam=='LGBM':
            xx,rel,sizes,seq,q=rank_queries(yy,x); self.model_=LGBMRanker(**self.c['models']['LGBM'])
            self.model_.fit(pd.DataFrame(xx,columns=self.pre_.names_),rel,group=sizes.tolist(),sample_weight=w[seq])
            self.query_audit_=q[KEYS].copy(); self.query_audit_['training_relevance']=rel; self.query_audit_['query_contiguous_row']=np.arange(len(q)); self.query_audit_['query_size']=q.groupby('unit_id').unit_id.transform('size'); self.query_audit_['sample_weight']=w[seq]
        else:
            cls={'RIDGE':Ridge,'EXTRATREES':ExtraTreesRegressor,'HISTGB':HistGradientBoostingRegressor}[fam]
            self.model_=cls(**self.c['models'][fam]); self.model_.fit(x,yy.target.to_numpy(float),sample_weight=w)
        self.training_groups_=sorted(set(yy.split_group)); return self
    def predict(self,f):
        x=self.pre_.transform(f)
        if self.spec['family']=='PAIRWISE': return self.model_.decision_function(x)
        return self.model_.predict(pd.DataFrame(x,columns=self.pre_.names_) if self.spec['family']=='LGBM' else x)

def score_groups(y,scores,ds):
    if scores.duplicated(['sample_id','model_id']).any(): raise ValueError('OOF or test duplicate prediction')
    rows=[]
    for (mid,scope,fold),s in scores.groupby(['model_id','scope','outer_fold'],sort=True):
        yy=y[y.sample_id.isin(s.sample_id)]
        if set(yy.sample_id)!=set(s.sample_id): raise ValueError('Scoring coverage mismatch')
        q=yy.merge(s,on=KEYS,validate='one_to_one')
        for unit,g in q.groupby('unit_id',sort=True):
            row={'dataset':ds,'scope':scope,'model_id':mid,'outer_fold':fold,'unit_id':unit,'split_group':str(g.split_group.iloc[0]),**group_metrics(g.target,g.score,g.sample_id)}
            rows.append(row)
    return pd.DataFrame(rows)

def random_groups(y,scope,ds,folds):
    rows=[]
    for unit,g in y.groupby('unit_id',sort=True):
        n=len(g); k=top_k(n); variable=g.target.nunique()>1
        rows.append({'dataset':ds,'scope':scope,'model_id':'RANDOM_EXPECTATION','outer_fold':int(folds[str(g.split_group.iloc[0])]),'unit_id':unit,'split_group':str(g.split_group.iloc[0]),'n':n,'top_k':k,
            'rho':0. if variable else np.nan,'top25_gain_pp':0.,'top25_overlap':k/n,'rho_random_variance':1/(n-1) if variable else np.nan,
            'top25_gain_random_variance':(1-k/n)*g.target.var(ddof=1)/k,'top25_overlap_random_variance':k*(k/n)*(1-k/n)*(n-k)/(n-1)/(k*k)})
    return pd.DataFrame(rows)

def resampling_units(metrics,ds):
    keys=['dataset','scope','model_id','split_group']
    if ds=='D1': return metrics[keys+METRICS].copy()
    return metrics.groupby(keys,sort=True)[METRICS].mean().reset_index()

def summarize(metrics,ds,c):
    units=resampling_units(metrics,ds); rows=[]; intervals=[]
    for (scope,mid),q in units.groupby(['scope','model_id'],sort=True):
        row={'dataset':ds,'scope':scope,'model_id':mid,'n_independent_units':len(q),'n_rho_defined':int(q.rho.notna().sum())}
        for m in METRICS:
            vals=q[m].dropna().to_numpy(); row[m]=float(vals.mean()) if len(vals) else np.nan; row[m+'_std_units']=float(vals.std(ddof=1)) if len(vals)>1 else np.nan
            if len(vals):
                rng=np.random.default_rng(SEED); boot=rng.choice(vals,size=(c['bootstrap']['replicates'],len(vals)),replace=True).mean(1)
                lo,hi=np.quantile(boot,[.025,.975]); row[m+'_ci95_low']=lo; row[m+'_ci95_high']=hi
        rows.append(row)
    return pd.DataFrame(rows),units

def contrasts(units,pairs_list,c):
    deltas=[]; summaries=[]
    for (ds,scope),q in units.groupby(['dataset','scope'],sort=True):
        available=set(q.model_id)
        for a,b in sorted(set(pairs_list)):
            if a==b or a not in available or b not in available: continue
            aa=q[q.model_id==a].set_index('split_group'); bb=q[q.model_id==b].set_index('split_group')
            for m in METRICS:
                diff=(aa[m]-bb[m]).dropna().sort_index(); vals=diff.to_numpy()
                for unit,v in diff.items(): deltas.append({'dataset':ds,'scope':scope,'method_a':a,'method_b':b,'metric':m,'split_group':unit,'delta':v})
                if not len(vals): continue
                rng=np.random.default_rng(SEED); boot=rng.choice(vals,size=(c['bootstrap']['replicates'],len(vals)),replace=True).mean(1)
                summaries.append({'dataset':ds,'scope':scope,'method_a':a,'method_b':b,'metric':m,'n_units_defined':len(vals),'mean_delta':vals.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),
                    'improved':int((vals>1e-12).sum()),'worsened':int((vals< -1e-12).sum()),'tied':int((abs(vals)<=1e-12).sum()),'min_unit_delta':vals.min(),'max_unit_delta':vals.max()})
    return pd.DataFrame(deltas),pd.DataFrame(summaries)
