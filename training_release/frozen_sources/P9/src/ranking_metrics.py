"""Ranking metrics for identical labeled patches. No prediction fitting here."""
import numpy as np
import pandas as pd

def top_k(n,fraction=.25,rounding='ceil'):
    if n<1: raise ValueError('Empty group')
    if not 0<fraction<=1: raise ValueError('Invalid fraction')
    return max(1,int(np.floor(n*fraction) if rounding=='floor' else np.ceil(n*fraction)))

def order(scores,ids):
    scores=np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite scores')
    if len(scores)!=len(ids) or len(set(ids))!=len(ids): raise ValueError('Nonunique or unequal IDs')
    return np.lexsort((np.asarray(ids,dtype=str),-scores))

def rho(y,s):
    y=np.asarray(y,dtype=float); s=np.asarray(s,dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(s).all(): raise ValueError('Nonfinite ranking input')
    if len(y)!=len(s) or len(y)<2: return np.nan
    a=pd.Series(y).rank(method='average').to_numpy(); b=pd.Series(s).rank(method='average').to_numpy()
    a-=a.mean(); b-=b.mean(); den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/den) if den>0 else np.nan

def group_metrics(y,s,ids,rounding='ceil'):
    y=np.asarray(y,dtype=float); s=np.asarray(s,dtype=float)
    if len(y)<4: raise ValueError('Ranking needs at least four labeled patches')
    pi=order(s,ids); ti=order(y,ids); k=top_k(len(y),rounding=rounding)
    ii,jj=np.triu_indices(len(y),1); dy=y[ii]-y[jj]; ds=s[ii]-s[jj]
    valid=dy!=0; concord=np.sign(dy[valid])*np.sign(ds[valid])
    confident=np.abs(dy)>=1.0
    return {'n':len(y),'top_k':k,'rho':rho(y,s),'top25_overlap':len(set(pi[:k])&set(ti[:k]))/k,
        'top25_gain_pp':float(y[pi[:k]].mean()-y.mean()),'top25_true_pce_mean':float(y[pi[:k]].mean()),'experiment_true_pce_mean':float(y.mean()),
        'top1_gain_pp':float(y[pi[:1]].mean()-y.mean()),'top4_gain_pp':float(y[pi[:min(4,len(y))]].mean()-y.mean()),
        'pair_concordance':float(np.mean((concord+1)/2)) if valid.any() else np.nan,
        'large_delta_ge1_concordance':float(np.mean((np.sign(dy[confident])*np.sign(ds[confident])+1)/2)) if confident.any() else np.nan,
        'small_delta_lt1_pair_count':int(((np.abs(dy)<1)&valid).sum()),'true_tied_pairs':int((~valid).sum()),
        'pred_tied_pairs':int((ds==0).sum()),'constant_prediction':len(np.unique(s))==1,'constant_target':len(np.unique(y))==1,
        'selected_low_0_to_2_count':int(((y[pi[:k]]>=0)&(y[pi[:k]]<=2)).sum()),
        'all_low_0_to_2_count':int(((y>=0)&(y<=2)).sum()),
        'true_boundary_tie_size':int((y==y[ti[k-1]]).sum()),'prediction_boundary_tie_size':int((s==s[pi[k-1]]).sum())}

def aggregate(experiment):
    keys=['scope','model_id']; fields=['rho','top25_overlap','top25_gain_pp','top1_gain_pp','top4_gain_pp','pair_concordance','large_delta_ge1_concordance']
    bydate=experiment.groupby(keys+['date'],sort=True)[fields].mean().reset_index()
    counts=experiment.groupby(keys+['date'],sort=True).agg(n_experiments=('expID','size'),n_rho_defined=('rho','count'),n_patches=('n','sum')).reset_index()
    bydate=counts.merge(bydate,on=keys+['date'],validate='one_to_one')
    summary=bydate.groupby(keys,sort=True)[fields].mean().reset_index()
    n=bydate.groupby(keys).agg(n_dates=('date','size'),n_experiments=('n_experiments','sum'),n_patches=('n_patches','sum')).reset_index()
    return bydate,n.merge(summary,on=keys,validate='one_to_one')
