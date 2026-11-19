"""P8 optical-only fusion, conservative correction and cross-fitted meta learner."""
import os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parents[1]
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[n]='1'
os.environ['MPLCONFIGDIR']=str(BASE/'.mplconfig'); sys.path.insert(0,str(ROOT/'research/pl_public_data_pilot_06_baselines_ablations/.deps'))
import json,hashlib,copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import p6_core as p6
import run_p6 as oldrun
KEYS=p6.KEYS; METAKEYS=['unit_id','split_group']; FEATURES=['score_spearman','top25_model_overlap','optical_robust_dispersion','base_boundary_margin','log_candidate_count']; METRICS=p6.METRICS+['top25_regret_pp']; SEED=20261020
sha=p6.sha; dump=p6.dump
def csv(df,path): df.to_csv(path,index=False,float_format='%.17g',lineterminator='\n')
def config(): return json.loads((BASE/'configs/frozen_p8_protocol.json').read_text(encoding='utf-8'))
def safe_dir(path):
    path=Path(path).resolve()
    if path==BASE or not path.is_relative_to(BASE): raise ValueError('P8 output boundary')
    return path
def group_hash(groups): return hashlib.sha256('\n'.join(sorted(map(str,groups))).encode()).hexdigest()
def id_hash(ids): return group_hash(ids)
def stats(ds):
    if ds not in ['D1','D2']: raise ValueError('Unknown dataset adapter')
    return p6.p5.STATS if ds=='D1' else p6.p4.F2
def split_groups(y,n=3):
    if y.split_group.nunique()<n: raise ValueError('INSUFFICIENT_GROUPS for grouped cross-fit')
    for i,(a,b) in enumerate(GroupKFold(n_splits=n).split(y,groups=y.split_group)):
        tr=set(y.iloc[a].split_group); va=set(y.iloc[b].split_group)
        if tr&va: raise ValueError('Group overlap')
        yield i,tr,va
def selected(scores,ids): return set(np.asarray(ids)[p6.order(scores,ids)[:p6.top_k(len(scores))]])
def rank_position_changes(base,score,ids):
    n=len(base); before=p6.order(base,ids); after=p6.order(score,ids); br=np.empty(n,int); ar=np.empty(n,int); br[before]=np.arange(n); ar[after]=np.arange(n)
    return int((br!=ar).sum())
def rank_scale(scores):
    s=np.asarray(scores,float)
    if s.ndim!=1 or not len(s) or not np.isfinite(s).all(): raise ValueError('Empty/nonfinite score vector')
    return (pd.Series(s).rank(method='average').to_numpy()-1)/(len(s)-1) if len(s)>1 else np.zeros(1)
def fuse_unit(base,aux,ids,weight,cap=False,normalization='RANK'):
    base=np.asarray(base,float); aux=np.asarray(aux,float); ids=np.asarray(ids,str); n=len(base)
    if len(aux)!=n or len(ids)!=n or len(set(ids))!=n: raise ValueError('Fusion identity mismatch')
    if not 0<=weight<=.5 or (cap and weight>.25): raise ValueError('Unfrozen lambda')
    if normalization not in ['RANK','RAW']: raise ValueError('Unknown normalization')
    b=rank_scale(base) if normalization=='RANK' else base.copy()
    if not np.isfinite(base).all(): raise ValueError('Invalid base: stop affected branch')
    status='SUCCESS'; requested=weight
    if not np.isfinite(aux).all(): a=b.copy(); weight=0.; status='AUX_NONFINITE_NOOP'
    else: a=rank_scale(aux) if normalization=='RANK' else aux.copy()
    if n==1: weight=0.; status='SINGLETON_NOOP'
    old=selected(base,ids); max_changes=int(np.floor(.25*p6.top_k(n))); attempts=0
    while True:
        score=(1-weight)*b+weight*a; changed=len(selected(score,ids)-old)
        if not cap or changed<=max_changes: break
        weight=weight/2 if attempts<16 else 0.; attempts+=1
    return score,{'lambda_requested':requested,'lambda_effective':weight,'changed_top25_members':changed,'changed_top25_fraction':changed/p6.top_k(n),'rank_positions_changed':rank_position_changes(base,score,ids),'cap_limit_members':max_changes if cap else -1,'cap_triggered':attempts>0,'status':status}
def gate_features(scores,optical,ds):
    p6.require(scores,KEYS+['base_score','aux_score']); p6.require(optical,KEYS+stats(ds)); rows=[]
    mean_cols=['mean'] if ds=='D1' else [s for s in stats(ds) if s.endswith('_mean') and not s.startswith('ND')]
    if not mean_cols: raise ValueError('No verified PL intensity adapter')
    for unit,g in scores.groupby('unit_id',sort=True):
        f=optical[optical.unit_id==unit]
        if not len(f) or not set(g.sample_id)<=set(f.sample_id): raise ValueError('Missing optical group context')
        b=g.base_score.to_numpy(); a=g.aux_score.to_numpy(); n=len(g); rank_scale(b); rank_scale(a)
        correlation=p6.rho(b,a); overlap=len(selected(b,g.sample_id)&selected(a,g.sample_id))/p6.top_k(n)
        x=f[mean_cols].to_numpy(); med=np.median(x,axis=0); mad=np.median(abs(x-med),axis=0); denom=np.maximum(np.median(abs(x),axis=0),1e-12); dispersion=float(np.median(np.clip(1.4826*mad/denom,0,10)))
        k=p6.top_k(n); ordered=b[p6.order(b,g.sample_id)]; scale=max(1.4826*np.median(abs(b-np.median(b))),1e-12); margin=float(np.clip((ordered[k-1]-ordered[k])/scale,0,10)) if k<n else 0.
        rows.append({'unit_id':unit,'split_group':str(g.split_group.iloc[0]),'score_spearman':float(correlation) if np.isfinite(correlation) else 0.,'top25_model_overlap':overlap,'optical_robust_dispersion':dispersion,'base_boundary_margin':margin,'log_candidate_count':np.log(n)})
    return pd.DataFrame(rows,columns=METAKEYS+FEATURES)
def require_meta(meta):
    if set(meta.columns)!=set(METAKEYS+FEATURES) or meta.unit_id.duplicated().any(): raise ValueError('Gate rejects targets, identities-as-features, or duplicate units')
    if not np.isfinite(meta[FEATURES]).all().all(): raise ValueError('Nonfinite gate feature')
def meta_weights(meta):
    n=meta.groupby('split_group').unit_id.transform('size').to_numpy(float); w=1/n; return w/w.mean()
def permutation_negative(meta,seed=SEED):
    require_meta(meta); z=meta.copy(deep=True); groups=sorted(meta.split_group.unique()); rng=np.random.default_rng(seed); seq=list(rng.permutation(groups)); mapping={seq[i]:seq[(i+1)%len(seq)] for i in range(len(seq))}
    if len(groups)<2: raise ValueError('Need at least2 independent groups for negative control')
    for dst,src in mapping.items():
        di=meta[meta.split_group==dst].sort_values('unit_id').index; sf=meta[meta.split_group==src].sort_values('unit_id')[FEATURES].to_numpy(); z.loc[di,FEATURES]=sf[np.arange(len(di))%len(sf)]
    return z,mapping
class MetaGate:
    def __init__(self,drop_disagreement=False,negative=False): self.drop=drop_disagreement; self.negative=negative
    def fit(self,meta,supervision):
        require_meta(meta)
        if set(supervision.columns)!=set(METAKEYS+['gate_target_delta']) or supervision.unit_id.duplicated().any(): raise ValueError('Wrong gate training target role')
        if set(meta.unit_id)!=set(supervision.unit_id): raise ValueError('Gate training identities differ')
        labels=meta[METAKEYS].merge(supervision,on=METAKEYS,validate='one_to_one').gate_target_delta.to_numpy(float)
        if not np.isfinite(labels).all(): raise ValueError('Nonfinite meta training target')
        self.training_groups_=set(meta.split_group); self.training_units_=set(meta.unit_id); self.names_=FEATURES[2:] if self.drop else FEATURES; self.permutation_={}; self.status_='SUCCESS'
        if len(self.training_groups_)<6: self.status_='INSUFFICIENT_GROUPS'
        elif np.max(abs(labels))<=1e-10: self.status_='NOOP_ZERO_TARGET'
        if self.negative: meta,self.permutation_=permutation_negative(meta)
        weights=meta_weights(meta); self.scaler_=StandardScaler().fit(meta[self.names_],sample_weight=weights)
        self.model_=Ridge(alpha=10.,fit_intercept=True,solver='svd').fit(self.scaler_.transform(meta[self.names_]),labels,sample_weight=weights) if self.status_=='SUCCESS' else None
        self.target_min_=float(labels.min()); self.target_max_=float(labels.max()); return self
    def predict(self,meta):
        require_meta(meta)
        if self.training_groups_&set(meta.split_group): raise ValueError('Gate inference group overlaps supervised gate training')
        if self.model_ is None: return np.zeros(len(meta))
        delta=self.model_.predict(self.scaler_.transform(meta[self.names_])); return .25*np.clip(delta/.5,0,1)
    def parameters(self):
        return {'status':self.status_,'feature_names':self.names_,'coef':self.model_.coef_.tolist() if self.model_ is not None else [],'intercept':float(self.model_.intercept_) if self.model_ is not None else 0.,'training_scaler_mean':self.scaler_.mean_.tolist(),'training_scaler_scale':self.scaler_.scale_.tolist(),'training_groups':sorted(self.training_groups_),'training_units_hash':id_hash(self.training_units_),'target_min':self.target_min_,'target_max':self.target_max_,'negative_permutation':self.permutation_}

def training_supervision(y,scores,allowed_training_groups):
    p6.require(y,KEYS+['target']); p6.require(scores,KEYS+['base_score','aux_score'])
    if set(y.split_group)!=set(allowed_training_groups) or set(y.sample_id)!=set(scores.sample_id): raise ValueError('Gate supervision must match its cross-fit training scope')
    q=y.merge(scores,on=KEYS,validate='one_to_one'); rows=[]
    for unit,g in q.groupby('unit_id',sort=True):
        blend,_=fuse_unit(g.base_score,g.aux_score,g.sample_id,.25)
        old=p6.group_metrics(g.target,g.base_score,g.sample_id)['top25_gain_pp']; new=p6.group_metrics(g.target,blend,g.sample_id)['top25_gain_pp']
        rows.append({'unit_id':unit,'split_group':str(g.split_group.iloc[0]),'gate_target_delta':new-old})
    return pd.DataFrame(rows,columns=METAKEYS+['gate_target_delta'])
def infer_policy(scores,meta,spec,gate=None):
    p6.require(scores,KEYS+['base_score','aux_score']); require_meta(meta)
    if set(meta.unit_id)!=set(scores.unit_id): raise ValueError('Inference meta/score unit mismatch')
    if spec['kind']=='LEARNED':
        if gate is None: raise ValueError('Missing fitted gate')
        values=gate.predict(meta); status=gate.status_
    elif spec['kind']=='RULE': values=np.where(meta.score_spearman.to_numpy()<=spec['threshold'],spec['weight'],0.); status='RULE_NO_LABELS'
    elif spec['kind'] in ['FIXED','STANDALONE']: values=np.full(len(meta),spec['weight']); status='FIXED_NO_LABELS'
    else: raise ValueError('Unsupported candidate kind')
    lookup=dict(zip(meta.unit_id,values)); frames=[]; annotations=[]
    for unit,g in scores.groupby('unit_id',sort=True):
        requested=float(lookup[unit]); requested=0. if len(g)<4 else requested
        if spec['kind']=='STANDALONE':
            final=g.aux_score.to_numpy(); rank_scale(final); _,audit=fuse_unit(g.base_score,g.aux_score,g.sample_id,0.); audit.update({'lambda_requested':np.nan,'lambda_effective':np.nan,'status':'STANDALONE_AUX','changed_top25_members':len(selected(final,g.sample_id)-selected(g.base_score,g.sample_id)),'rank_positions_changed':rank_position_changes(g.base_score,final,g.sample_id)}); audit['changed_top25_fraction']=audit['changed_top25_members']/p6.top_k(len(g))
        else: final,audit=fuse_unit(g.base_score,g.aux_score,g.sample_id,requested,spec.get('cap',False),spec.get('normalization','RANK'))
        z=g.copy(); z['final_score']=final; chosen=selected(final,g.sample_id); z['selected_top25']=z.sample_id.isin(chosen); z['base_selected_top25']=z.sample_id.isin(selected(g.base_score,g.sample_id)); z['lambda_g']=audit['lambda_effective']; z['current_target_used_for_inference']=False; z['auxiliary']=spec['aux']; frames.append(z)
        annotations.append({'unit_id':unit,'split_group':str(g.split_group.iloc[0]),'gate_status':status,**audit,'n_candidates':len(g),'top_k':p6.top_k(len(g)),'auxiliary':spec['aux'],'cap_enabled':spec.get('cap',False),'normalization':spec.get('normalization','RANK')})
    return pd.concat(frames,ignore_index=True),pd.DataFrame(annotations)
def evaluate(y,pred,ds,mid,scope,fold):
    p6.require(y,KEYS+['target'])
    if set(y.sample_id)!=set(pred.sample_id) or pred.sample_id.duplicated().any(): raise ValueError('Evaluation candidate pool changed')
    q=y.merge(pred[KEYS+['final_score']],on=KEYS,validate='one_to_one'); rows=[]
    for unit,g in q.groupby('unit_id',sort=True):
        mt=p6.group_metrics(g.target,g.final_score,g.sample_id); oracle=g.target.to_numpy()[p6.order(g.target,g.sample_id)[:p6.top_k(len(g))]].mean(); mt['top25_regret_pp']=float(oracle-mt['top25_true_pce_mean'])
        rows.append({'dataset':ds,'scope':scope,'model_id':mid,'outer_fold':fold,'unit_id':unit,'split_group':str(g.split_group.iloc[0]),**mt})
    return pd.DataFrame(rows)
def independent_metrics(mt): return mt.groupby(['dataset','scope','model_id','split_group'],sort=True)[METRICS].mean().reset_index()
def tail_values(values):
    v=np.asarray(values,float)
    if not len(v) or not np.isfinite(v).all(): raise ValueError('Incomplete tail sample')
    n=max(1,int(np.ceil(.2*len(v)))); return {'worst_delta_pp':float(v.min()),'q10_delta_pp':float(np.quantile(v,.1)),'cvar20_delta_pp':float(np.sort(v)[:n].mean()),'cvar_group_count':n,'negative_groups':int((v< -1e-10).sum()),'severe_groups':int((v< -.5).sum()),'improved_groups':int((v>1e-10).sum()),'tied_groups':int((abs(v)<=1e-10).sum())}
def candidate_summary(mt,specs):
    u=independent_metrics(mt); base=u[u.model_id=='NOOP'].set_index('split_group'); ss={s['model_id']:s for s in specs}; rows=[]
    for mid,g in u.groupby('model_id',sort=True):
        q=g.set_index('split_group'); diff=(q.top25_gain_pp-base.top25_gain_pp).to_numpy(); spec=ss[mid]
        rows.append({'model_id':mid,'wave':spec['wave'],'kind':spec['kind'],'auxiliary':spec['aux'],'eligible':spec['eligible'],'complexity':spec['complexity'],'n_independent_groups':len(g),**{m:float(g[m].mean()) for m in METRICS},'mean_delta_pp':float(diff.mean()),**tail_values(diff)})
    return pd.DataFrame(rows)
def choose(summary,model_ids=None):
    z=summary[summary.eligible.astype(bool)].copy(); base=summary.loc[summary.model_id=='NOOP','top25_gain_pp'].iloc[0]
    if model_ids is not None: z=z[z.model_id.isin(set(model_ids)|{'NOOP'})]
    z=z[(z.top25_gain_pp>base+1e-10)|(z.model_id=='NOOP')]
    for name in ['top25_gain_pp','q10_delta_pp','rho']: z[name+'_sort']=z[name].round(10)
    z=z.sort_values(['top25_gain_pp_sort','q10_delta_pp_sort','severe_groups','rho_sort','complexity','model_id'],ascending=[False,False,True,False,True,True],na_position='last')
    return str(z.iloc[0].model_id)
def feature_scores(bank,aux):
    col='POS_score' if aux=='POS' else 'TREE_score'; return bank[KEYS+['base_score',col]].rename(columns={col:'aux_score'}).copy()
def diagnostic_specs(specs):
    out=[]
    for aux in ['POS','TREE']:
        base=copy.deepcopy(next(s for s in specs if s['model_id']==f'LEARNED_{aux}'))
        for suffix,flag in [('NO_DISAGREE','drop_disagreement'),('NEGATIVE_META','negative')]:
            s=copy.deepcopy(base); s.update({'model_id':f'{suffix}_{aux}','eligible':False,'wave':'DIAGNOSTIC',flag:True}); out.append(s)
        s=copy.deepcopy(next(s for s in specs if s['model_id']==f'FIX_{aux}_25')); s.update({'model_id':f'RAW_FIXED_{aux}_25','normalization':'RAW','eligible':False,'wave':'DIAGNOSTIC'}); out.append(copy.deepcopy(s))
        s.update({'model_id':f'AUX_{aux}_ALONE','kind':'STANDALONE','normalization':'RAW','weight':0.}); out.append(copy.deepcopy(s))
    return out
