"""D1-only features, grouped fitting and scoring. No IO during prediction."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import hashlib,json,re,warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import ConvergenceWarning
from ranking_metrics import group_metrics,rho,order,top_k

BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parents[1]; SEED=20261019
os.environ['MPLCONFIGDIR']=str(BASE/'.mplconfig')
KEYS=['sample_id','substrateID','patchID']
STATS=['mean','std','early10_mean','late10_mean','late_minus_early','maximum','argmax_relative_index','relative_end_change','middle50_mean','relative_index_slope','integral_frame_index','total_abs_difference']
PAIR_MODELS=['ID_PAIRWISE','PL_PAIRWISE','ID_PLUS_PL_PAIRWISE']
SCORED_MODELS=['PL_INTENSITY','ID_PRIOR']+PAIR_MODELS+['AUTHOR_KNN_AUX']
METRICS=['rho','top25_gain_pp','top25_overlap']

def config(): return json.loads((BASE/'configs/frozen_p5_d1_protocol.json').read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def dump(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def csv(df,path): df.to_csv(path,index=False,float_format='%.12g',lineterminator='\n')
def safe_dir(path):
    path=Path(path).resolve()
    if path==BASE or not path.is_relative_to(BASE): raise ValueError('Output must be a child of P5')
    return path

def canonical_patch(v):
    if isinstance(v,(bool,np.bool_)): raise ValueError('Boolean patchID')
    if isinstance(v,(int,np.integer)): n=int(v)
    elif isinstance(v,str) and re.fullmatch(r'[0-9]+',v): n=int(v)
    else: raise ValueError('Nonintegral/ambiguous patchID')
    if n<1: raise ValueError('Nonpositive patchID')
    return n

def identity(raw):
    sub=raw.substrateID
    if sub.isna().any() or not sub.map(lambda v:isinstance(v,str) and bool(v.strip()) and v==v.strip()).all(): raise ValueError('Invalid substrate identity')
    q=pd.DataFrame({'substrateID':sub.to_numpy(),'patchID':[canonical_patch(v) for v in raw.patchID]})
    q['sample_id']=[f'D1::{s}::{p:02d}' for s,p in zip(q.substrateID,q.patchID)]
    if q.sample_id.duplicated().any(): raise ValueError('Duplicate substrate+patch identity')
    return q[KEYS]

def optical(raw):
    q=identity(raw); q['LP725']=raw.LP725.to_numpy()
    return q

def target(raw):
    q=identity(raw); q['pce_mean']=pd.to_numeric(raw.pce_mean,errors='raise').to_numpy(float)
    if not np.isfinite(q.pce_mean).all(): raise ValueError('Nonfinite D1 target: preserve source and block main comparison')
    return q

def require(frame,columns):
    if set(frame.columns)!=set(columns): raise ValueError('Optical/ID prediction API rejects targets and extra fields')
    if frame.sample_id.duplicated().any(): raise ValueError('Duplicate sample identity')

def clean_signal(x):
    x=np.asarray(x,dtype=float)
    if x.shape!=(719,): raise ValueError('D1 complete LP725 must have 719 frames')
    finite=np.isfinite(x)
    if finite.all(): return x.copy()
    if not finite.any(): return np.zeros(719)
    idx=np.arange(719); return np.interp(idx,idx[finite],x[finite])

def raw_signals(opt):
    require(opt,KEYS+['LP725'])
    return np.vstack([clean_signal(x) for x in opt.LP725])

def features(opt):
    a=raw_signals(opt); n=a.shape[1]; w=int(np.ceil(.1*n)); early=a[:,:w].mean(1); late=a[:,-w:].mean(1)
    t=np.linspace(0,1,n); centered=t-t.mean()
    vals=np.column_stack([a.mean(1),a.std(1),early,late,late-early,a.max(1),a.argmax(1)/(n-1),
        (late-early)/np.maximum(abs(early),1e-12),a[:,int(np.floor(.25*n)):int(np.ceil(.75*n))].mean(1),
        a@centered/(centered@centered),np.trapezoid(a,dx=1,axis=1),np.abs(np.diff(a,axis=1)).sum(1)])
    if not np.isfinite(vals).all(): raise ValueError('Nonfinite statistics after frozen repair')
    return pd.concat([opt[KEYS].reset_index(drop=True),pd.DataFrame(vals,columns=STATS)],axis=1)

class Relative:
    def fit(self,f):
        require(f,KEYS+STATS); self.floor_=np.maximum(1e-12,.01*np.median(abs(f[STATS].to_numpy(float)),axis=0)); return self
    def transform(self,f):
        require(f,KEYS+STATS)
        med=f.groupby('substrateID',sort=False)[STATS].transform('median')
        mad=(f[STATS]-med).abs().groupby(f.substrateID,sort=False).transform('median')
        return (f[STATS].to_numpy(float)-med.to_numpy())/np.maximum(1.4826*mad.to_numpy(),self.floor_)

def one_hot(f,c):
    return (f.patchID.to_numpy()[:,None]==np.asarray(c['categories'])[None,:]).astype(float)

class IDPrior:
    def fit(self,y):
        require(y,KEYS+['pce_mean']); z=y.copy(); z['residual']=z.pce_mean-z.groupby('substrateID').pce_mean.transform('mean')
        self.table_=z.groupby('patchID').residual.agg(['sum','count','mean']).reset_index(); self.table_['score']=self.table_['sum']/(self.table_['count']+5)
        self.lookup_=self.table_.set_index('patchID').score; return self
    def predict(self,ids):
        require(ids,KEYS); return ids.patchID.map(self.lookup_).fillna(0).to_numpy(float)

def construct_pairs(y,x,tolerance=.5):
    require(y,KEYS+['pce_mean']); blocks=[]; audits=[]; identities=[]
    y=y.reset_index(drop=True)
    for sub,g in y.groupby('substrateID',sort=True):
        g=g.sort_values('sample_id'); idx=g.index.to_numpy(); i,j=np.triu_indices(len(g),1); dy=g.pce_mean.to_numpy()[i]-g.pce_mean.to_numpy()[j]
        keep=abs(dy)>tolerance; aa=idx[i[keep]]; bb=idx[j[keep]]
        audits.append({'substrateID':sub,'n_labeled':len(g),'unordered_pairs_all':len(i),'unordered_pairs_used':len(aa),'excluded_near_tie_training_only':int((~keep).sum())})
        if len(aa):
            blocks.append((sub,x[aa]-x[bb],(dy[keep]>0).astype(int)))
            identities.extend(f'{y.loc[a,"sample_id"]}|{y.loc[b,"sample_id"]}' for a,b in zip(aa,bb))
    if not blocks: raise ValueError('No informative training pairs')
    total=sum(2*len(x[1]) for x in blocks); xx=[]; yy=[]; ww=[]
    for sub,diff,label in blocks:
        xx.extend([diff,-diff]); yy.extend([label,1-label]); ww.extend([np.full(len(diff),total/(len(blocks)*2*len(diff)))]*2)
    return np.concatenate(xx),np.concatenate(yy),np.concatenate(ww),pd.DataFrame(audits),identities

class Pairwise:
    def __init__(self,c,model_id):
        if model_id not in PAIR_MODELS: raise ValueError('Unapproved model')
        self.c=c; self.mid=model_id
    def block(self,f):
        if self.mid=='ID_PAIRWISE': require(f,KEYS); return one_hot(f,self.c)
        require(f,KEYS+STATS); pl=np.column_stack([f[STATS].to_numpy(),self.rel_.transform(f)])
        return pl if self.mid=='PL_PAIRWISE' else np.column_stack([one_hot(f,self.c),pl])
    def fit(self,y,f):
        self.rel_=Relative().fit(f) if self.mid!='ID_PAIRWISE' else None
        self.names_=([f'patchID_{v}' for v in self.c['categories']] if self.mid!='PL_PAIRWISE' else [])+([f'{kind}_{v}' for kind in ['absolute','relative'] for v in STATS] if self.mid!='ID_PAIRWISE' else [])
        inp=f[KEYS] if self.mid=='ID_PAIRWISE' else f
        self.scaler_=StandardScaler().fit(self.block(inp)); x=pd.DataFrame(self.scaler_.transform(self.block(inp)),index=f.sample_id)
        ys=y.sort_values(['substrateID','sample_id']).reset_index(drop=True)
        xx,yy,ww,self.audit_,self.pair_ids_=construct_pairs(ys,x.loc[ys.sample_id].to_numpy(),self.c['pairwise']['tolerance_pp'])
        self.pair_hash_=hashlib.sha256('\n'.join(self.pair_ids_).encode()).hexdigest(); self.n_pairs_=len(yy)
        self.model_=LogisticRegression(random_state=SEED,**self.c['pairwise']['estimator'])
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning); self.model_.fit(xx,yy,sample_weight=ww)
        self.weights_sum_=float(ww.sum()); return self
    def predict(self,f): return self.model_.decision_function(self.scaler_.transform(self.block(f)))

def substrate_metrics(y,scores,model_id,scope,fold):
    if not set(y.sample_id)==set(scores.sample_id): raise ValueError('Scoring candidate identity mismatch')
    q=y.merge(scores,on=KEYS,validate='one_to_one'); rows=[]
    for sub,g in q.groupby('substrateID',sort=True):
        result=group_metrics(g.pce_mean,g.score,g.sample_id)
        rows.append({'scope':scope,'outer_fold':fold,'model_id':model_id,'substrateID':sub,**result})
    return pd.DataFrame(rows)

def choose_rule(y,f,folds,c):
    rows=[]
    for inner in sorted(folds.foldNo.unique()):
        ids=set(folds.loc[folds.foldNo==inner,'sample_id']); iy=y[y.sample_id.isin(ids)]; fi=f[f.sample_id.isin(ids)]
        for name in c['simple_rules']['candidates']:
            for direction in c['simple_rules']['directions']:
                rid=name+('_PLUS' if direction==1 else '_MINUS'); sc=fi[KEYS].copy(); sc['score']=direction*fi[name].to_numpy()
                mt=substrate_metrics(iy,sc,rid,'INNER_TRAIN_ONLY',int(inner)); rows.append(mt)
    allmetrics=pd.concat(rows,ignore_index=True)
    ranking=allmetrics.groupby('model_id')[['rho','top25_gain_pp']].mean().reset_index().sort_values(['rho','top25_gain_pp','model_id'],ascending=[False,False,True],na_position='last')
    rid=ranking.iloc[0].model_id; name,suffix=rid.rsplit('_',1)
    return (name,1 if suffix=='PLUS' else -1),rid,allmetrics

def choose_knn(y,opt,folds,c):
    x=raw_signals(opt); ids=opt.sample_id.to_numpy(); ys=y.set_index('sample_id').loc[ids].pce_mean.to_numpy(); ff=folds.set_index('sample_id').loc[ids].foldNo.to_numpy()
    rows=[]
    # Neighbors of the largest k contain all candidates; identical uniform-distance kNN.
    for fold in sorted(set(ff)):
        tr=ff!=fold; va=~tr; ks=c['author_knn_aux']['neighbors_candidates']; model=KNeighborsRegressor(n_neighbors=max(ks),weights='uniform',n_jobs=1,algorithm='brute').fit(x[tr],ys[tr])
        distances,neighbors=model.kneighbors(x[va]); cumulative=np.cumsum(ys[tr][neighbors],axis=1)
        for k in ks: rows.append({'inner_fold':int(fold),'k':k,'MAE':float(np.mean(abs(ys[va]-cumulative[:,k-1]/k))),'n_inner_train':int(tr.sum()),'n_inner_validation':int(va.sum())})
    detail=pd.DataFrame(rows); rank=detail.groupby('k').MAE.mean().reset_index().sort_values(['MAE','k']); return int(rank.iloc[0].k),detail

def random_metrics(y,scope,fold):
    rows=[]
    for sub,g in y.groupby('substrateID',sort=True):
        n=len(g); k=top_k(n); variable=g.pce_mean.nunique()>1
        rows.append({'scope':scope,'outer_fold':fold,'model_id':'RANDOM_EXPECTATION','substrateID':sub,'n':n,'top_k':k,
            'rho':0. if variable else np.nan,'top25_gain_pp':0.,'top25_overlap':k/n,'constant_target':not variable,'constant_prediction':False,
            'rho_random_variance':1/(n-1) if variable else np.nan,'top25_gain_random_variance':(1-k/n)*g.pce_mean.var(ddof=1)/k,
            'top25_overlap_random_variance':k*(k/n)*(1-k/n)*(n-k)/(n-1)/(k*k)})
    return pd.DataFrame(rows)

def bootstrap(metrics,c):
    macro=[]; boot=[]; delta=[]; pairs=[]
    for (scope,mid),q in metrics.groupby(['scope','model_id'],sort=True):
        row={'scope':scope,'model_id':mid,'n_substrates':len(q),'n_patches':int(q.n.sum()),'rho_defined':int(q.rho.notna().sum())}
        for m in METRICS:
            vals=q[m].dropna().to_numpy(); row[m]=float(vals.mean()) if len(vals) else np.nan
            if len(vals):
                rng=np.random.default_rng(SEED); bs=rng.choice(vals,size=(c['bootstrap']['replicates'],len(vals)),replace=True).mean(1)
                boot.append({'scope':scope,'kind':'ABSOLUTE','method_a':mid,'method_b':'NONE','metric':m,'n_substrates_defined':len(vals),'mean':vals.mean(),'ci95_low':np.quantile(bs,.025),'ci95_high':np.quantile(bs,.975),'seed':SEED})
        macro.append(row)
    for scope,q in metrics.groupby('scope',sort=True):
        for a,b in c['contrasts']:
            aa=q[q.model_id==a].set_index('substrateID'); bb=q[q.model_id==b].set_index('substrateID')
            for m in METRICS:
                diff=(aa[m]-bb[m]).dropna().sort_index(); vals=diff.to_numpy()
                for sub,v in diff.items(): delta.append({'scope':scope,'method_a':a,'method_b':b,'metric':m,'substrateID':sub,'delta':v})
                if not len(vals): continue
                rng=np.random.default_rng(SEED); bs=rng.choice(vals,size=(c['bootstrap']['replicates'],len(vals)),replace=True).mean(1)
                row={'scope':scope,'kind':'PAIRED','method_a':a,'method_b':b,'metric':m,'n_substrates_defined':len(vals),'mean':vals.mean(),'ci95_low':np.quantile(bs,.025),'ci95_high':np.quantile(bs,.975),'improved':int((vals>1e-12).sum()),'worsened':int((vals< -1e-12).sum()),'tied':int((abs(vals)<=1e-12).sum()),'seed':SEED}
                boot.append(row); pairs.append(row)
    return pd.DataFrame(macro),pd.DataFrame(boot),pd.DataFrame(delta),pd.DataFrame(pairs)
