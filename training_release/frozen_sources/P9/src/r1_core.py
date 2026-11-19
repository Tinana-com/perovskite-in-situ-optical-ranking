"""Frozen P9 R1 representations, compiled RNG seeding and original pair loss."""
import os,sys,time,json,hashlib,warnings
from pathlib import Path
import r0_core as r
BASE=r.BASE; ROOT=r.ROOT; R1=BASE/'R1'; KEYS=r.KEYS
os.environ['NUMBA_CACHE_DIR']=str(R1/'.numba_cache'); os.environ['MPLCONFIGDIR']=str(R1/'.mplconfig')
sys.path.insert(0,str(BASE/'sources/minirocket'))
import minirocket as mr
import numpy as np
import pandas as pd
from numba import njit,set_num_threads
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
set_num_threads(1)

@njit(cache=True)
def seed_compiled(seed): np.random.seed(seed)

def array_hash(a):
    x=np.ascontiguousarray(a); return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def combined_hash(xs): return hashlib.sha256('\n'.join(array_hash(a) for a in xs).encode()).hexdigest()
def id_hash(ids): return hashlib.sha256('\n'.join(map(str,ids)).encode()).hexdigest()

def prepare_optical(ds,opt):
    channels=r.config()['M2']['channels'][ds]; allowed=KEYS+(['LP725'] if ds=='D1' else r.CHANNELS)
    r.p.require(opt,allowed); arrays={}; counters=[]
    for ch in channels:
        rows=[]
        for sid,trace in zip(opt.sample_id,opt[ch]):
            z,status=r.resample_shape(trace); rows.append(z); counters.append({'sample_id':sid,'channel':ch,'raw_length':len(trace),**status})
        arrays[ch]=np.ascontiguousarray(np.vstack(rows),dtype=np.float32)
    return arrays,pd.DataFrame(counters)

def shape_features(arrays,channels):
    u=np.linspace(0,1,256); w=u[:64]; w=w-w.mean(); late=u[192:]; late=late-late.mean(); block=[]; names=[]
    for ch in channels:
        z=np.asarray(arrays[ch],dtype=np.float64); d=np.diff(z,axis=1)
        block.append(np.column_stack([z[:,192:]@late/(late@late)-z[:,:64]@w/(w@w),np.mean(np.diff(z,n=2,axis=1)**2,axis=1),np.mean(d>1e-8,axis=1),np.mean(d< -1e-8,axis=1)]))
        names.extend(ch+'_'+s for s in ['local_slope_contrast','curvature_energy','positive_step_fraction','negative_step_fraction'])
    x=np.column_stack(block); assert np.isfinite(x).all(); return x,names

class Rocket:
    def __init__(self,ds): self.ds=ds; self.channels=r.config()['M2']['channels'][ds]
    def fit(self,arrays,train_idx,ids,groups):
        self.parameters={}; self.training_ids_=list(map(str,np.asarray(ids)[train_idx])); self.training_groups_=sorted(set(np.asarray(groups)[train_idx])); self.training_input_hashes_={}
        for ch in self.channels:
            x=np.ascontiguousarray(arrays[ch][train_idx],dtype=np.float32); seed_compiled(r.config()['seed'])
            self.parameters[ch]=mr.fit(x,num_features=1008,max_dilations_per_kernel=32); self.training_input_hashes_[ch]=array_hash(x)
        self.parameter_hash_=combined_hash([v for ch in self.channels for v in self.parameters[ch]]); return self
    def transform(self,arrays,indices):
        blocks=[mr.transform(np.ascontiguousarray(arrays[ch][indices],dtype=np.float32),self.parameters[ch]) for ch in self.channels]
        x=np.ascontiguousarray(np.column_stack(blocks),dtype=np.float64)
        if x.shape!=(len(indices),1008*len(self.channels)) or not np.isfinite(x).all(): raise ValueError('MiniRocket feature shape/finiteness')
        return x
    def save(self,path):
        np.savez_compressed(path,**{ch+'__'+name:v for ch in self.channels for name,v in zip(['dilations','num_features_per_dilation','biases'],self.parameters[ch])})

def split_inputs(ds,features,labels,validation_groups):
    r.p.require(features,KEYS+(r.p.p5.STATS if ds=='D1' else r.p.p4.F2)); r.p.require(labels,KEYS+['target'])
    val=set(map(str,validation_groups)); tridx=np.flatnonzero(~features.split_group.isin(val).to_numpy()); evidx=np.flatnonzero(features.split_group.isin(val).to_numpy())
    yy=labels[~labels.split_group.isin(val)].sort_values(['split_group','unit_id','sample_id']).reset_index(drop=True)
    if not len(tridx) or not len(evidx) or val&set(yy.split_group) or set(features.iloc[tridx].unit_id)&set(features.iloc[evidx].unit_id): raise ValueError('Invalid train/validation split')
    if not set(yy.sample_id)<=set(features.iloc[tridx].sample_id): raise ValueError('Training labels not within optical training input')
    return tridx,evidx,yy

def fit_ranker(ds,train_x,train_meta,yy):
    """P6 exact scaler/pairs/weights/loss; feature matrix changes only."""
    start=time.perf_counter(); train_x=np.asfortranarray(train_x,dtype=np.float64); scaler=StandardScaler().fit(train_x); scaled=scaler.transform(train_x)
    lookup=pd.DataFrame(scaled,index=train_meta.sample_id); x=lookup.loc[yy.sample_id].to_numpy()
    xx,label,weight,audit,pair_ids=r.p.pairs(yy,x,ds); del scaled,lookup,x
    clf=LogisticRegression(random_state=20261019,**r.config()['pairwise']['parameters'])
    with warnings.catch_warnings(record=True) as logs:
        warnings.simplefilter('always',ConvergenceWarning); clf.fit(xx,label,sample_weight=weight)
    warning_text=[str(w.message) for w in logs]; converged=not any(issubclass(w.category,ConvergenceWarning) for w in logs)
    info={'fit_status':'SUCCESS' if converged else 'NOT_STABLE_CONVERGENCE','warnings':warning_text,'solver_iterations':int(clf.n_iter_[0]),'fit_wall_seconds':time.perf_counter()-start,'n_ordered_pairs':len(label),'pair_ID_sha256':id_hash(pair_ids),'pair_weight_sum':float(weight.sum()),'pair_matrix_bytes':int(xx.nbytes),'pair_weight_hash':array_hash(weight),'pair_labels_hash':array_hash(label),'pair_matrix_shape':list(xx.shape),'n_features':train_x.shape[1]}
    return scaler,clf,info,audit

def fit_case(ds,mid,features,labels,val_groups,arrays,shapes,rocket=None):
    """Public training entry strips outer-validation labels before all fitting."""
    if mid not in ['M0','M1','M2','M3']: raise ValueError('Frozen four models only')
    tr,ev,yy=split_inputs(ds,features,labels,val_groups); cols=r.p.p5.STATS if ds=='D1' else r.p.p4.F2
    if mid=='M0': x=features[cols].to_numpy(float); names=list(cols); tx=x[tr]; vx=x[ev]
    elif mid=='M1': x=np.column_stack([features[cols].to_numpy(float),shapes]); names=list(cols)+shape_features(arrays,r.config()['M2']['channels'][ds])[1]; tx=x[tr]; vx=x[ev]
    else:
        if rocket is None: rocket=Rocket(ds).fit(arrays,tr,features.sample_id.to_numpy(),features.split_group.to_numpy())
        expected=set(features.iloc[tr].sample_id)
        if set(rocket.training_ids_)!=expected or set(rocket.training_groups_)&set(map(str,val_groups)): raise ValueError('Rocket fitted outside current training groups')
        tx=rocket.transform(arrays,tr); vx=rocket.transform(arrays,ev); names=[ch+f'_ROCKET_{i:04d}' for ch in rocket.channels for i in range(1008)]
        if mid=='M3': tx=np.column_stack([features.iloc[tr][cols].to_numpy(float),tx]); vx=np.column_stack([features.iloc[ev][cols].to_numpy(float),vx]); names=list(cols)+names
    scaler,model,info,audit=fit_ranker(ds,tx,features.iloc[tr],yy); before=time.perf_counter(); decision=model.decision_function(scaler.transform(vx)); infer=time.perf_counter()-before
    scores=features.iloc[ev][KEYS].copy().reset_index(drop=True); scores['score']=decision
    if not np.isfinite(decision).all(): raise ValueError('Nonfinite decision score')
    info.update({'dataset':ds,'model_id':mid,'train_optical_rows':len(tr),'validation_optical_rows':len(ev),'train_labeled_rows':len(yy),'training_groups':sorted(set(yy.split_group)),'validation_groups':sorted(map(str,val_groups)),'training_sample_ID_sha256':id_hash(features.iloc[tr].sample_id),'validation_sample_ID_sha256':id_hash(features.iloc[ev].sample_id),'scaler_mean_sha256':array_hash(scaler.mean_),'scaler_scale_sha256':array_hash(scaler.scale_),'coefficient_sha256':array_hash(model.coef_),'infer_wall_seconds':infer,'rocket_parameter_sha256':rocket.parameter_hash_ if mid in ['M2','M3'] else 'NOT_USED','validation_target_used':False,'feature_names':names})
    fitted={'scaler':scaler,'classifier':model,'info':info,'pair_audit':audit,'rocket':rocket,'train_x':tx if len(tx)<200 else None}
    return scores,fitted

def save_fitted(path,fitted):
    s=fitted['scaler']; model=fitted['classifier']; np.savez_compressed(path,scaler_mean=s.mean_,scaler_scale=s.scale_,coefficient=model.coef_,intercept=model.intercept_,n_iter=model.n_iter_)

def rotation_allowed(old_metadata,remote_sha,local_sha,new_verified,reviewed_and_R1_authorized):
    return bool(new_verified and reviewed_and_R1_authorized and remote_sha and remote_sha==local_sha
        and old_metadata.get('id')=='1_Onraxzpb3qByjfxwJv6u6eTTq39PZl8'
        and old_metadata.get('title')=='PL_P9_HANDOFF_R0.zip'
        and old_metadata.get('mime_type')=='application/zip'
        and old_metadata.get('parent_ids')==['1eWdLYQwDe8huQMJfgbDLpoNOOpej7EDx'])
