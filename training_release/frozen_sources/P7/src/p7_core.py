"""Dual-view, Top-K/listwise and smooth group-CVaR; train-only supervision."""
import os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'research/pl_public_data_pilot_06_baselines_ablations/.deps'))
for s in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[s]='1'
os.environ['MPLCONFIGDIR']=str(BASE/'.mplconfig')
import copy,hashlib,json,warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
import p6_core as p6
import run_p6 as oldrun

KEYS=p6.KEYS; SEED=20261019; METRICS=p6.METRICS+['top25_regret_pp']
sha=p6.sha; csv=p6.csv; dump=p6.dump; safe_dir=p6.safe_dir
def config(): return json.loads((BASE/'configs/frozen_p7_protocol.json').read_text(encoding='utf-8'))
def stats(ds): return p6.p5.STATS if ds=='D1' else p6.p4.F2
def views(): return config()['W1_views']
def modules(): return config()['W2_modules']
def view(name): return copy.deepcopy(next(v for v in views() if v['view_id']==name))
def module(name): return copy.deepcopy(next(v for v in modules() if v['module_id']==name))

class DualView:
    def __init__(self,ds,view_spec,c): self.ds=ds; self.v=copy.deepcopy(view_spec); self.c=c
    def fit(self,f):
        p6.require(f,KEYS+stats(self.ds)); self.names_=[]; self.floor_=np.maximum(1e-12,.01*np.median(abs(f[stats(self.ds)].to_numpy()),axis=0))
        if self.v['position']: self.names_ += ['POS_'+str(i) for i in self.c['categories']]
        if self.v['absolute']: self.names_ += ['ABS_'+s for s in stats(self.ds)]
        if self.v['relative']: self.names_ += ['Z_'+s for s in stats(self.ds)]+['RANK_'+s for s in stats(self.ds)]
        self.scaler_=StandardScaler().fit(self.raw_block(f)); self.gate_=np.array([self.v['position_gate'] if n.startswith('POS_') else self.v['relative_gate'] if n.startswith(('Z_','RANK_')) else 1. for n in self.names_]); return self
    def raw_block(self,f):
        p6.require(f,KEYS+stats(self.ds)); parts=[]; x=f[stats(self.ds)]
        if self.v['position']: parts.append((f.patchID.to_numpy()[:,None]==np.asarray(self.c['categories'])[None,:]).astype(float))
        if self.v['absolute']: parts.append(x.to_numpy(float))
        if self.v['relative']:
            med=f.groupby('unit_id',sort=False)[stats(self.ds)].transform('median'); mad=(x-med).abs().groupby(f.unit_id,sort=False).transform('median')
            parts.append((x.to_numpy()-med.to_numpy())/np.maximum(1.4826*mad.to_numpy(),self.floor_))
            ranks=f.groupby('unit_id',sort=False)[stats(self.ds)].rank(method='average'); n=f.groupby('unit_id').unit_id.transform('size').to_numpy()[:,None]
            parts.append(np.where(n>1,(ranks.to_numpy()-1)/np.maximum(n-1,1)-.5,0))
        block=np.column_stack(parts)
        if not np.isfinite(block).all(): raise ValueError('Nonfinite dual-view input')
        return block
    def transform(self,f): return self.scaler_.transform(self.raw_block(f))*self.gate_

def perturb_features(f,ds,gain):
    if gain not in [.95,1.,1.05]: raise ValueError('Amplitude drift exceeds frozen bound')
    p6.require(f,KEYS+stats(ds)); z=f.copy(deep=True); scalable=[s for s in stats(ds) if s not in ['argmax_relative_index','relative_end_change']]
    z[scalable]=z[scalable]*gain; return z

def shuffle_negative_control(f,ds):
    p6.require(f,KEYS+stats(ds)); z=f.copy(deep=True); groups=sorted(f.split_group.unique()); ranked=sorted(groups,key=lambda g:hashlib.sha256(f'{SEED}|group|{g}'.encode()).hexdigest()); chosen=set(ranked[:max(1,len(ranked)//2)])
    for unit,g in f[f.split_group.isin(chosen)].groupby('unit_id',sort=True):
        seed=int(hashlib.sha256(f'{SEED}|shuffle|{unit}'.encode()).hexdigest()[:16],16); rng=np.random.default_rng(seed); idx=g.sort_values('sample_id').index.to_numpy(); z.loc[idx,stats(ds)]=f.loc[idx[rng.permutation(len(idx))],stats(ds)].to_numpy()
    return z,chosen

def top_flags(y):
    flags={}
    for unit,g in y.groupby('unit_id',sort=True):
        k=p6.top_k(len(g)); good=set(g.sort_values(['target','sample_id'],ascending=[False,True]).sample_id.iloc[:k]); flags.update({s:s in good for s in g.sample_id})
    return flags

def weighted_pairs(y,x,ds,boost):
    if boost not in [1.,2.,4.]: raise ValueError('Unfrozen TopK boost')
    xx,labels,ww,audit,ids=p6.pairs(y,x,ds); original=ww.copy(); flags=top_flags(y); unitcol='substrateID' if ds=='D1' else 'expID'; pairptr=0; rowptr=0; codes=[]; group_names=sorted(y.split_group.unique()); lookup={g:i for i,g in enumerate(group_names)}
    for a in audit.to_dict('records'):
        n=int(a['unordered_pairs_used'] if ds=='D1' else a['usable_unordered_pairs']); unit=a[unitcol]
        if not n: continue
        mult=np.array([boost if flags[l]!=flags[r] else 1. for l,r in (s.split('|') for s in ids[pairptr:pairptr+n])]); mult/=mult.mean()
        ww[rowptr:rowptr+2*n]*=np.tile(mult,2); group=y.loc[y.unit_id==unit,'split_group'].iloc[0]; codes.extend([lookup[group]]*(2*n)); pairptr+=n; rowptr+=2*n
    if rowptr!=len(labels) or pairptr!=len(ids): raise ValueError('Pair identity / row mapping mismatch')
    if not np.isclose(ww.sum(),original.sum()): raise ValueError('TopK weighting changed total ERM mass')
    return xx,labels,ww,np.asarray(codes,int),group_names,audit,ids

class TrainObjective:
    """Exact analytic derivatives; validation/test outcomes never accepted here."""
    def __init__(self,y,xcopies,ds,m):
        self.y=y; self.m=copy.deepcopy(m); blocks=[]; labels=[]; weights=[]; codes=[]; original_count=None
        for x in xcopies:
            xx,yy,ww,cc,names,audit,ids=weighted_pairs(y,x,ds,m['topk_boost']); blocks.append(xx); labels.append(yy); weights.append(ww/len(xcopies)); codes.append(cc)
            if original_count is None: original_count=len(yy); self.names=names; self.audit=audit; self.ids=ids
            elif self.ids!=ids: raise ValueError('Augmentation changed supervised pairs')
        self.Xpair=np.concatenate(blocks); self.labels=np.concatenate(labels); self.ww=np.concatenate(weights); self.codes=np.concatenate(codes); self.Z=self.Xpair*(2*self.labels-1)[:,None]; self.G=len(self.names); self.den=np.bincount(self.codes,weights=self.ww,minlength=self.G); self.original_pairs=original_count
        self.Xpoint=np.concatenate(xcopies); self.dimension=self.Z.shape[1]; self.group_names=self.names
        unitstarts=np.flatnonzero(y.unit_id.ne(y.unit_id.shift()).to_numpy()); sizes=np.diff(np.r_[unitstarts,len(y)]); unitgroups=np.array([self.names.index(g) for g in y.split_group.iloc[unitstarts]])
        unit_counts=np.bincount(unitgroups,minlength=self.G); targets=[]
        for start,n in zip(unitstarts,sizes):
            v=y.target.iloc[start:start+n].to_numpy(); k=p6.top_k(n); ordered=np.sort(v)[::-1]; boundary=.5*(ordered[k-1]+ordered[k]); q=expit((v-boundary)/.5); targets.append(q/q.sum())
        q=np.concatenate(targets); self.soft_target=np.tile(q,len(xcopies)); self.starts=np.concatenate([unitstarts+i*len(y) for i in range(len(xcopies))]); self.sizes=np.tile(sizes,len(xcopies)); self.unit_groups=np.tile(unitgroups,len(xcopies)); self.unit_weights=1/(np.tile(unit_counts[unitgroups],len(xcopies))*len(xcopies)); self.row_unit=np.repeat(np.arange(len(self.starts)),self.sizes)
        self.pair_hash=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    def losses(self,w):
        margin=self.Z@w; pairloss=np.logaddexp(0,-margin); pl=np.bincount(self.codes,weights=self.ww*pairloss,minlength=self.G)/self.den
        if self.m['listwise_lambda']:
            score=self.Xpoint@w; maxima=np.maximum.reduceat(score,self.starts); centered=score-np.repeat(maxima,self.sizes); ee=np.exp(centered); sums=np.add.reduceat(ee,self.starts); prob=ee/np.repeat(sums,self.sizes); logprob=centered-np.repeat(np.log(sums),self.sizes)
            unitloss=-np.add.reduceat(self.soft_target*logprob,self.starts); ll=np.bincount(self.unit_groups,weights=self.unit_weights*unitloss,minlength=self.G)
        else: ll=np.zeros(self.G); prob=None
        lam=self.m['listwise_lambda']; return (1-lam)*pl+lam*ll,margin,prob,pl,ll
    def value_grad(self,theta):
        w=theta[:self.dimension]; L,margin,prob,pl,ll=self.losses(w); eta=self.m['cvar_eta']; q=self.m['cvar_q']
        if eta:
            tau=theta[-1]; residual=(L-tau)/.05; tail=expit(residual); value=(1-eta)*L.mean()+eta*(tau+.05*np.logaddexp(0,residual).sum()/(q*self.G)); group_grad=((1-eta)+eta/q*tail)/self.G; grad_tau=eta*(1-tail.sum()/(q*self.G))
        else: value=L.mean(); group_grad=np.full(self.G,1/self.G); grad_tau=None
        lam=self.m['listwise_lambda']; v=-(1-lam)*group_grad[self.codes]*(self.ww/self.den[self.codes])*expit(-margin); grad=self.Z.T@v
        if lam:
            rowweights=group_grad[self.unit_groups][self.row_unit]*self.unit_weights[self.row_unit]
            grad+=lam*(self.Xpoint.T@(rowweights*(prob-self.soft_target)))
        value+=.5*np.dot(w,w)/self.original_pairs; grad+=w/self.original_pairs
        return float(value),np.r_[grad,grad_tau] if eta else grad
    def diagnostic(self,theta,stage):
        L,_,_,pl,ll=self.losses(theta[:self.dimension]); eta=self.m['cvar_eta']; tail=expit((L-theta[-1])/.05) if eta else np.zeros(self.G); factor=(1-eta)+eta/self.m['cvar_q']*tail
        return pd.DataFrame({'training_split_group':self.names,'stage':stage,'total_group_loss':L,'pair_logistic_loss':pl,'listwise_loss':ll,'effective_risk_factor':factor,'cvar_eta':eta,'cvar_q':self.m['cvar_q'],'tau':theta[-1] if eta else np.nan})

class NotStable(RuntimeError): pass
class DVR:
    def __init__(self,ds,v,m,c): self.ds=ds; self.v=copy.deepcopy(v); self.m=copy.deepcopy(m); self.c=c
    def fit(self,y,f,negative=False):
        p6.require(y,KEYS+['target']); p6.require(f,KEYS+stats(self.ds))
        if not set(y.sample_id)<=set(f.sample_id): raise ValueError('Training source identity mismatch')
        self.training_groups_=set(y.split_group); self.pre=DualView(self.ds,self.v,self.c).fit(f)
        if negative: f,self.shuffled_groups_=shuffle_negative_control(f,self.ds)
        else: self.shuffled_groups_=set()
        yy=y.sort_values(['split_group','unit_id','sample_id']).reset_index(drop=True); copies=[]
        gains=[1.,.95,1.05] if self.m['amplitude_augmentation'] else [1.]
        for gain in gains:
            fp=perturb_features(f,self.ds,gain); lookup=pd.DataFrame(self.pre.transform(fp),index=f.sample_id); copies.append(lookup.loc[yy.sample_id].to_numpy())
        obj=TrainObjective(yy,copies,self.ds,self.m); warm=LogisticRegression(random_state=SEED,**self.c['ordinary_estimator'])
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            try: warm.fit(obj.Xpair,obj.labels,sample_weight=obj.ww)
            except ConvergenceWarning as exc: raise NotStable(str(exc)) from exc
        theta=warm.coef_[0].copy()
        if self.m['cvar_eta']: theta=np.r_[theta,np.quantile(obj.losses(theta)[0],1-self.m['cvar_q'])]
        initial=theta.copy(); custom=bool(self.m['cvar_eta'] or self.m['listwise_lambda'])
        if custom:
            options={k:v for k,v in self.c['custom_solver'].items() if k!='method'}; result=minimize(obj.value_grad,theta,jac=True,method='L-BFGS-B',options=options); theta=result.x
            self.solver_status_=str(result.message); self.iterations_=int(result.nit)
            if not result.success or not np.isfinite(result.fun): raise NotStable('custom objective NOT_STABLE: '+str(result.message))
        else: self.solver_status_='SKLEARN_LBFGS_CONVERGED'; self.iterations_=int(warm.n_iter_[0])
        self.coef_=theta[:obj.dimension]; self.parameter_count_=len(theta); self.feature_count_=obj.dimension; self.pair_hash_=obj.pair_hash; self.pair_count_=obj.original_pairs; self.training_pair_mass_=float(obj.ww.sum()); self.copies_=len(gains)
        self.group_diagnostic_=pd.concat([obj.diagnostic(initial,'INITIAL'),obj.diagnostic(theta,'FINAL')],ignore_index=True); self.objective_=float(obj.value_grad(theta)[0]); self.gradient_inf_=float(abs(obj.value_grad(theta)[1]).max()); self.pair_audit_=obj.audit
        return self
    def predict(self,f): return self.pre.transform(f)@self.coef_

def grouped_metrics(y,score,ds,method,scope,outer_fold):
    q=y.merge(score,on=KEYS,validate='one_to_one'); rows=[]
    if len(q)!=len(y): raise ValueError('Shared candidate coverage changed')
    for unit,g in q.groupby('unit_id',sort=True):
        mt=p6.group_metrics(g.target,g.score,g.sample_id); k=p6.top_k(len(g)); oracle=g.target.to_numpy()[p6.order(g.target,g.sample_id)[:k]].mean(); mt['top25_regret_pp']=float(oracle-mt['top25_true_pce_mean'])
        rows.append({'dataset':ds,'scope':scope,'model_id':method,'outer_fold':outer_fold,'unit_id':unit,'split_group':str(g.split_group.iloc[0]),**mt})
    return pd.DataFrame(rows)
def unit_metrics(mt,ds):
    return mt.groupby(['dataset','scope','model_id','split_group'],sort=True)[METRICS].mean().reset_index()
def macro(mt,ds): return unit_metrics(mt,ds).groupby(['dataset','scope','model_id'],sort=True)[METRICS].mean().reset_index()
def inner_splits(y):
    for i,(a,b) in enumerate(GroupKFold(n_splits=3).split(y,groups=y.split_group)):
        tr=set(y.iloc[a].split_group); va=set(y.iloc[b].split_group)
        if tr&va: raise ValueError('Inner group leakage')
        yield i,tr,va

def select_view(summary):
    z=summary[summary.status=='SUCCESS'].sort_values(['top25_gain_pp','rho','top25_overlap','n_features','view_id'],ascending=[False,False,False,True,True]);
    if not len(z): raise ValueError('No stable W1 view')
    return str(z.iloc[0].view_id)
def select_module(own,other,own_baseline,other_baseline):
    a=own[own.status=='SUCCESS'].set_index('module_id'); b=other[other.status=='SUCCESS'].set_index('module_id'); rows=[]
    for mid in sorted(set(a.index)&set(b.index)):
        da=(a.loc[mid,'top25_gain_pp']-own_baseline)/max(abs(own_baseline),.1); db=(b.loc[mid,'top25_gain_pp']-other_baseline)/max(abs(other_baseline),.1); mm=module(mid)
        complexity=int(mm['topk_boost']>1)+int(mm['listwise_lambda']>0)+int(mm['cvar_eta']>0)+int(mm['amplitude_augmentation'])
        rows.append({'module_id':mid,'own_relative_gain':da,'other_relative_gain':db,'min_relative_gain':min(da,db),'mean_relative_gain':.5*(da+db),'active_modules':complexity})
    rank=pd.DataFrame(rows).sort_values(['min_relative_gain','mean_relative_gain','active_modules','module_id'],ascending=[False,False,True,True])
    if not len(rank): raise ValueError('No stable joint W2 module')
    return str(rank.iloc[0].module_id),rank
