"""P4: fixed rules and ONE new within-experiment pairwise linear ranker."""
from __future__ import annotations
import os
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parents[1]
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
os.environ['MPLCONFIGDIR']=str(BASE/'.mplconfig')
import argparse
import hashlib
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
from ranking_metrics import group_metrics,aggregate,top_k,order,rho
from audit_p3 import audit as audit_p3

CHANNELS=['LP725','LP780','SP775','ND']
F2=[f'{ch}_{stat}' for ch in CHANNELS for stat in ['mean','last','slope']]
RULES=[f'{ch}_{stat}' for ch in CHANNELS for stat in ['mean','last','end_start_delta']]
KEYS=['sample_id','date','expID','patchID']; GROUP=['date','expID']; SEED=20261019

def config(): return json.loads((BASE/'configs/frozen_p4_protocol.json').read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def dump(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def csv(df,path): df.to_csv(path,index=False,float_format='%.12g',lineterminator='\n')
def safe_dir(path):
    path=Path(path).resolve()
    if path==BASE or not path.is_relative_to(BASE): raise ValueError('Writes restricted to P4')
    return path

def optical(frame):
    if frame.sample_id.duplicated().any() or frame.duplicated(['date','expID','patchID']).any(): raise ValueError('Duplicate identity')
    if frame.groupby('expID').date.nunique().max()>1: raise ValueError('Experiment reused across dates')
    if not np.isfinite(frame[F2].to_numpy(float)).all(): raise ValueError('Nonfinite optics')
    return frame[KEYS+F2].copy()

def context(opt):
    # Adapted directly from P3 optical/group_context/Relative; original SHA256 in inventory.
    x=opt[KEYS+F2].copy(); med=x.groupby(GROUP,sort=False)[F2].transform('median')
    mad=(x[F2]-med).abs().groupby([x.date,x.expID],sort=False).transform('median')
    return x,med.to_numpy(float),mad.to_numpy(float)

class Relative:
    def fit(self,opt):
        self.floor_=np.maximum(1e-12,.01*np.median(abs(opt[F2].to_numpy(float)),axis=0)); return self
    def transform(self,opt):
        x,med,mad=context(opt)
        return (x[F2].to_numpy(float)-med)/np.maximum(1.4826*mad,self.floor_)

def construct_pairs(train,x,tolerance=.5):
    """All unordered training pairs, both orientations; date/experiment balanced."""
    train=train.reset_index(drop=True)
    blocks=[]; audits=[]
    for (date,exp),group in train.groupby(GROUP,sort=True):
        group=group.sort_values('sample_id'); idx=group.index.to_numpy()
        ii,jj=np.triu_indices(len(idx),1); dy=group.PCE_backward.to_numpy(float)[ii]-group.PCE_backward.to_numpy(float)[jj]
        keep=np.abs(dy)>=tolerance; i=idx[ii[keep]]; j=idx[jj[keep]]; target=(dy[keep]>0).astype(int)
        auditrow={'date':int(date),'expID':exp,'n_labeled':len(group),'all_unordered_pairs':len(ii),'usable_unordered_pairs':len(i),
            'near_tie_pairs_excluded_from_training_only':int((~keep).sum()),'n_low_0_to_2_training_patches':int(((group.PCE_backward>=0)&(group.PCE_backward<=2)).sum())}
        audits.append(auditrow)
        if len(i): blocks.append((int(date),exp,x[i]-x[j],target,i,j))
    if not blocks: raise ValueError('No informative training pairs; do not fabricate preferences')
    experiments_by_date={d:sum(b[0]==d for b in blocks) for d in sorted({b[0] for b in blocks})}
    nd=len(experiments_by_date); total=sum(2*len(b[2]) for b in blocks)
    xx=[]; yy=[]; ww=[]; pairids=[]
    for date,exp,diff,target,ii,jj in blocks:
        xx.extend([diff,-diff]); yy.extend([target,1-target])
        w=total/(nd*experiments_by_date[date]*2*len(diff)); ww.extend([np.full(len(diff),w)]*2)
        pairids.extend([f'{train.loc[i,"sample_id"]}|{train.loc[j,"sample_id"]}' for i,j in zip(ii,jj)])
    return np.concatenate(xx),np.concatenate(yy),np.concatenate(ww),pd.DataFrame(audits),pairids

class PairRanker:
    def __init__(self,c): self.c=c
    def fit(self,train_labels,train_opt):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        self.relative_=Relative().fit(train_opt); self.scaler_=StandardScaler()
        allx=self.scaler_.fit_transform(self.relative_.transform(train_opt))
        lookup=pd.DataFrame(allx,index=train_opt.sample_id)
        train=train_labels.sort_values(['date','expID','sample_id']).reset_index(drop=True)
        x=lookup.loc[train.sample_id].to_numpy()
        xx,yy,ww,self.pair_audit_,self.pair_ids_=construct_pairs(train,x,self.c['pairwise_ranker']['pair_tolerance_pp'])
        self.model_=LogisticRegression(random_state=SEED,**self.c['pairwise_ranker']['estimator'])
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            self.model_.fit(xx,yy,sample_weight=ww)
        self.n_pairs_=len(yy); self.pair_weight_sum_=float(ww.sum()); self.training_dates_=sorted(train.date.unique().tolist())
        return self
    def predict(self,eval_opt):
        if set(eval_opt.columns)!=set(KEYS+F2): raise ValueError('Prediction accepts identifiers and optical values only, never PCE/label-presence')
        x=self.scaler_.transform(self.relative_.transform(eval_opt))
        return self.model_.decision_function(x)

def inner_folds(dates):
    dates=sorted(map(int,dates))
    if len(dates)%2: raise ValueError('Even training date count required')
    for i in range(0,len(dates),2): yield i//2,[d for d in dates if d not in dates[i:i+2]],dates[i:i+2]

def select_naive(train,c,allowed,outer):
    if set(train.date)!=set(allowed) or set(allowed)&set(c['holdout_dates']): raise ValueError('Inner selector may use only declared outer training dates')
    rows=[]
    for fold,_,va in inner_folds(allowed):
        sub=train[train.date.isin(va)]
        for feature in RULES:
            for (date,exp),q in sub.groupby(GROUP):
                if len(q)<4: continue
                m=group_metrics(q.PCE_backward,q[feature],q.sample_id)
                rows.append({'outer_fold':outer,'inner_fold':fold,'date':int(date),'expID':exp,'rule':feature,'rho':m['rho'],'top25_gain_pp':m['top25_gain_pp']})
    detail=pd.DataFrame(rows)
    bydate=detail.groupby(['rule','date'])[['rho','top25_gain_pp']].mean().reset_index()
    score=bydate.groupby('rule')[['rho','top25_gain_pp']].mean().reset_index()
    score=score.sort_values(['rho','top25_gain_pp','rule'],ascending=[False,False,True],na_position='last')
    if score.rho.notna().sum()==0: raise ValueError('All naive rules undefined on training dates')
    choice=score.iloc[0].rule; score['outer_fold']=outer
    return choice,detail,score

def detail_predictions(meta,scores,model,scope,fold,regression=False):
    z=meta[KEYS+['PCE_backward']].rename(columns={'PCE_backward':'y_true'}).copy()
    z['score']=np.asarray(scores,float); z['model_id']=model; z['scope']=scope; z['outer_fold']=fold; z['seed']=SEED
    z['score_is_PCE_prediction']=regression
    return z

def score_groups(pred):
    metrics=[]; patches=[]
    for (scope,mid,date,exp),q in pred.groupby(['scope','model_id']+GROUP,sort=True):
        q=q.sort_values('sample_id').copy()
        m=group_metrics(q.y_true,q.score,q.sample_id)
        metrics.append(m|{'scope':scope,'model_id':mid,'date':int(date),'expID':exp,'seed':SEED,
            'PCE_mae_pp':float(abs(q.y_true-q.score).mean()) if q.score_is_PCE_prediction.iloc[0] else np.nan})
        pi=order(q.score,q.sample_id); ti=order(q.y_true,q.sample_id); k=m['top_k']
        positions=np.empty(len(q),int); positions[pi]=np.arange(1,len(q)+1)
        trpos=np.empty(len(q),int); trpos[ti]=np.arange(1,len(q)+1)
        q['predicted_order_stable']=positions; q['true_order_stable_for_scoring']=trpos
        q['predicted_average_rank']=(-q.score).rank(method='average'); q['true_average_rank_for_scoring']=(-q.y_true).rank(method='average')
        q['selected_top25']=q.predicted_order_stable<=k; q['true_top25_for_scoring']=q.true_order_stable_for_scoring<=k
        patches.append(q)
    return pd.DataFrame(metrics),pd.concat(patches,ignore_index=True)

def random_metrics(meta,c):
    theory=[]; monte=[]; repdate=[]; fixed=[]
    for (scope,date,exp),q in meta.groupby(['scope']+GROUP,sort=True):
        q=q.sort_values('sample_id'); y=q.PCE_backward.to_numpy(float); n=len(q); k=top_k(n)
        seed=int.from_bytes(hashlib.sha256(f'{SEED}|{scope}|{date}|{exp}'.encode()).digest()[:8],'little')
        rng=np.random.default_rng(seed); u=rng.random((c['random']['replicates'],n))
        idx=np.argsort(-u,axis=1,kind='stable'); top=idx[:,:k]; truth=order(y,q.sample_id)[:k]
        ranks=np.argsort(np.argsort(u,axis=1),axis=1)+1; a=pd.Series(y).rank().to_numpy(); a-=a.mean()
        den=np.sqrt(np.dot(a,a)*np.sum((ranks-ranks.mean(axis=1,keepdims=True))**2,axis=1))
        rv=np.full(len(u),np.nan); np.divide((ranks-ranks.mean(axis=1,keepdims=True))@a,den,out=rv,where=den>0)
        overlap=np.isin(top,truth).mean(axis=1); gain=y[top].mean(axis=1)-y.mean()
        t1=y[idx[:,0]]-y.mean(); t4=y[idx[:,:min(4,n)]].mean(axis=1)-y.mean()
        base={'scope':scope,'date':int(date),'expID':exp,'n':n,'top_k':k,'seed':SEED}
        theory.append(base|{'model_id':'RANDOM_EXPECTATION','rho':0. if np.unique(y).size>1 else np.nan,'top25_overlap':k/n,'top25_gain_pp':0.,'top1_gain_pp':0.,'top4_gain_pp':0.,'pair_concordance':.5 if np.unique(y).size>1 else np.nan,'large_delta_ge1_concordance':.5 if (y.max()-y.min())>=1 else np.nan})
        monte.append(base|{'model_id':'RANDOM_MONTE_CARLO_MEAN','rho':float(np.nanmean(rv)) if np.isfinite(rv).any() else np.nan,'top25_overlap':float(overlap.mean()),'top25_gain_pp':float(gain.mean()),'top1_gain_pp':float(t1.mean()),'top4_gain_pp':float(t4.mean()),'pair_concordance':np.nan,'large_delta_ge1_concordance':np.nan})
        repdate.append(pd.DataFrame({'scope':scope,'date':int(date),'expID':exp,'replicate':np.arange(len(u)),'rho':rv,'top25_overlap':overlap,'top25_gain_pp':gain}))
        fixed.append(detail_predictions(q,u[0],'RANDOM_FIXED',scope,-1))
    random_rep=pd.concat(repdate,ignore_index=True).groupby(['scope','date','replicate'])[['rho','top25_overlap','top25_gain_pp']].mean().reset_index()
    macro_rep=random_rep.groupby(['scope','replicate'])[['rho','top25_overlap','top25_gain_pp']].mean().reset_index()
    return pd.DataFrame(theory+monte),random_rep,macro_rep,pd.concat(fixed,ignore_index=True)

def paired(bydate,c):
    details=[]; summaries=[]
    for scope,d in bydate.groupby('scope'):
        base=d[d.model_id=='NAIVE_INNER_SELECTED'].set_index('date')
        for mid,part in d.groupby('model_id'):
            if mid.startswith('NAIVE_') or mid.startswith('RANDOM_'): continue
            part=part.set_index('date')
            for metric in ['rho','top25_overlap','top25_gain_pp']:
                delta=(part[metric]-base[metric]).dropna(); vals=delta.to_numpy(float)
                if not len(vals): continue
                rng=np.random.default_rng(SEED); boot=rng.choice(vals,size=(10000,len(vals)),replace=True).mean(axis=1)
                for date,v in delta.items(): details.append({'scope':scope,'model_id':mid,'baseline':'NAIVE_INNER_SELECTED','metric':metric,'date':int(date),'delta':v,'seed':SEED})
                summaries.append({'scope':scope,'model_id':mid,'baseline':'NAIVE_INNER_SELECTED','metric':metric,'mean_delta':float(vals.mean()),
                    'improved_dates':int((vals>1e-12).sum()),'worsened_dates':int((vals< -1e-12).sum()),'tied_dates':int((abs(vals)<=1e-12).sum()),
                    'ci95_low':float(np.quantile(boot,.025)),'ci95_high':float(np.quantile(boot,.975)),'seed':SEED})
    return pd.DataFrame(details),pd.DataFrame(summaries)

def inventory(out,c):
    paths=[ROOT/c['data_path']]
    for key in ['p1_path','p2_path','p3_path']:
        paths += [x for x in (ROOT/c[key]).rglob('*') if x.is_file() and not any(part.startswith(('.venv','.test_tmp','.mplconfig','__pycache__','.pytest_cache')) for part in x.parts)]
    rows=[{'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path),'version':c['git_head']+'; untracked exact source identified by SHA256','access':'READ_ONLY'} for path in sorted(set(paths))]
    csv(pd.DataFrame(rows),out/'source_inventory.csv')

def load_inputs(c):
    src=ROOT/c['data_path']
    if not src.is_file(): raise FileNotFoundError(f'Missing real D2: {src}; restore existing file; no simulation or automatic download')
    if src.stat().st_size!=c['data_bytes'] or sha(src)!=c['data_sha256']: raise ValueError('D2 SHA256 mismatch')
    feature_path=ROOT/c['p2_path']/'results/extracted_features.csv'
    if f'{sha(feature_path)}  results/extracted_features.csv' not in (ROOT/c['p2_path']/'SHA256_MANIFEST.txt').read_text(encoding='utf-8'): raise ValueError('P2 feature source mismatch')
    f=pd.read_csv(feature_path); optical(f)
    if len(f)!=11136 or f.PCE_backward.notna().sum()!=5857: raise ValueError('D2 feature count mismatch')
    # Exact original identity and labels, not invented arrays or inferred outcomes.
    raw=pd.read_hdf(src,key='df')
    for col in ['date','patchID']:
        if not np.array_equal(pd.to_numeric(raw[col]).to_numpy(int),pd.to_numeric(f[col]).to_numpy(int)): raise ValueError('Raw D2 identity mismatch')
    if not np.array_equal(raw.expID.astype(str).to_numpy(),f.expID.astype(str).to_numpy()): raise ValueError('Raw experiment mismatch')
    if not np.allclose(raw.PCE_backward,f.PCE_backward,equal_nan=True,atol=1e-10): raise ValueError('Raw label mismatch')
    previous={}
    for scope,name in [(c['development_scope'],'outer_oof_predictions.csv'),(c['holdout_scope'],'heldout_predictions.csv')]:
        path=ROOT/c['p3_path']/'results'/name
        if f'{sha(path)}  results/{name}' not in (ROOT/c['p3_path']/'SHA256_MANIFEST.txt').read_text(encoding='utf-8'): raise ValueError('Published P3 predictions modified')
        previous[scope]=pd.read_csv(path)
    # P3 did not output F2 Ridge in its historical CSV; exact same P2 14-date model is frozen evidence.
    p2h=pd.read_csv(ROOT/c['p2_path']/'results/frozen_model_heldout_predictions.csv')
    p2h=p2h[p2h.model_id=='F2_ridge'].copy(); p2h['model_id']='F2_ridge'; p2h['outer_fold']=-1
    previous[c['holdout_scope']]=pd.concat([previous[c['holdout_scope']],p2h],ignore_index=True)
    return f,previous

def run(out):
    out=safe_dir(out)
    if out.exists() and any(out.iterdir()): raise ValueError('Existing run protected; use new path')
    r=out/'results'; r.mkdir(parents=True); c=config(); start=time.perf_counter(); cpu=time.process_time()
    audit_p3(r); inventory(r,c); f,previous=load_inputs(c)
    eligible=f.groupby(GROUP).PCE_backward.count().rename('n_labels').reset_index()
    eligible['eligible']=eligible.n_labels>=4; eligible['reason']=np.where(eligible.eligible,'AT_LEAST_4_FINITE_LABELS','INSUFFICIENT_LABELS_ONLY')
    eligible=eligible.merge(f.groupby(GROUP).size().rename('n_optical').reset_index(),on=GROUP)
    csv(eligible,r/'experiment_eligibility.csv')
    eligible_keys=set(map(tuple,eligible.loc[eligible.eligible,GROUP].to_numpy()))
    evaluated=f[f.PCE_backward.notna()&f[GROUP].apply(tuple,axis=1).isin(eligible_keys)].copy()
    evaluated['scope']=np.where(evaluated.date.isin(c['development_dates']),c['development_scope'],c['holdout_scope'])
    rows=[]; choices=[]; inner=[]; scores=[]; pa=[]; coeff=[]; foldrows=[]
    from threadpoolctl import threadpool_limits
    folds=[(i,[d for d in c['development_dates'] if d not in va],va,c['development_scope']) for i,va in enumerate(c['outer_validation_dates'])]
    folds.append((-1,c['development_dates'],c['holdout_dates'],c['holdout_scope']))
    with threadpool_limits(limits=1):
        for fold,trdates,va,scope in folds:
            tr=f[f.date.isin(trdates)&f.PCE_backward.notna()]; ev=evaluated[evaluated.date.isin(va)]
            tr_opt=optical(f[f.date.isin(trdates)]); ev_opt=optical(f[f.date.isin(va)])
            if set(tr_opt.date)&set(ev_opt.date) or set(tr_opt.expID)&set(ev_opt.expID): raise ValueError('Train/evaluation group overlap')
            choice,detail,score=select_naive(tr,c,trdates,fold); choices.append({'outer_fold':fold,'scope':scope,'rule':choice,'training_dates':';'.join(map(str,trdates)),'seed':SEED})
            inner.append(detail); scores.append(score)
            ranker=PairRanker(c).fit(tr,tr_opt); allpred=pd.Series(ranker.predict(ev_opt),index=ev_opt.sample_id)
            rows.append(detail_predictions(ev,allpred.loc[ev.sample_id],'PAIRWISE_LOGISTIC_REL12',scope,fold))
            for rule in RULES: rows.append(detail_predictions(ev,ev[rule],f'NAIVE_{rule}',scope,fold))
            rows.append(detail_predictions(ev,ev[choice],'NAIVE_INNER_SELECTED',scope,fold))
            for mid in c['existing_models']:
                ref=previous[scope][previous[scope].model_id==mid].set_index('sample_id')
                aligned=ref.loc[ev.sample_id]
                if not np.allclose(aligned.y_true,ev.PCE_backward,atol=1e-10): raise ValueError('Frozen P3 targets/samples differ')
                rows.append(detail_predictions(ev,aligned.y_pred,f'P3_{mid}',scope,fold,True))
            pair_audit=ranker.pair_audit_.copy(); pair_audit['outer_fold']=fold; pa.append(pair_audit)
            pair_hash=hashlib.sha256('\n'.join(ranker.pair_ids_).encode()).hexdigest()
            for j,name in enumerate(F2): coeff.append({'outer_fold':fold,'scope':scope,'feature':name,'coefficient':float(ranker.model_.coef_[0,j]),
                'relative_floor_training_only':float(ranker.relative_.floor_[j]),'scaler_mean_training_only':float(ranker.scaler_.mean_[j]),'scaler_scale_training_only':float(ranker.scaler_.scale_[j]),
                'ordered_pair_count':ranker.n_pairs_,'pair_weight_sum':ranker.pair_weight_sum_,'unordered_pair_ids_sha256':pair_hash,'solver_iterations':int(ranker.model_.n_iter_[0])})
            for d in trdates+va: foldrows.append({'outer_fold':fold,'scope':scope,'date':d,'role':'validation' if d in va else 'train'})
            print(f'{scope} fold={fold} train={len(tr)} eval={len(ev)} strongest_training_rule={choice} pairs={ranker.n_pairs_}',flush=True)
            if time.perf_counter()-start>c['wall_budget_seconds']: raise TimeoutError('4-hour budget reached; preserve results')
    random_exp,random_date,random_macro,fixed=random_metrics(evaluated,c)
    pred=pd.concat(rows+[fixed],ignore_index=True); exp,patch=score_groups(pred)
    exp=pd.concat([exp,random_exp],ignore_index=True)
    date,summary=aggregate(exp)
    csv(patch,r/'patch_scores_and_ranks.csv'); csv(exp,r/'ranking_metrics_by_experiment.csv'); csv(date,r/'ranking_metrics_by_date.csv'); csv(summary,r/'ranking_summary_date_macro.csv')
    csv(exp.groupby(['scope','model_id']).agg(n_experiments=('expID','size'),rho=('rho','mean'),positive_rho_fraction=('rho',lambda x:(x>0).mean()),top25_gain_pp=('top25_gain_pp','mean')).reset_index(),r/'ranking_summary_experiment_equal.csv')
    csv(random_date,r/'random_simulation_date_replicates.csv'); csv(random_macro,r/'random_simulation_macro_replicates.csv')
    rd,ps=paired(date,c); csv(rd,r/'paired_date_deltas.csv'); csv(ps,r/'paired_summary.csv')
    csv(pd.DataFrame(choices),r/'naive_rule_choices.csv'); csv(pd.concat(inner),r/'naive_inner_validation_metrics.csv'); csv(pd.concat(scores),r/'naive_inner_selection_scores.csv')
    csv(pd.concat(pa),r/'training_pair_audit.csv'); csv(pd.DataFrame(coeff),r/'ranker_coefficients_and_preprocessing.csv'); csv(pd.DataFrame(foldrows),r/'fold_assignments.csv')
    # Every negative-gain case and every negative/undefined rho is retained, not just a favorable subset.
    failed=exp[(exp.top25_gain_pp<0)|(exp.rho<0)|exp.rho.isna()].copy(); csv(failed,r/'all_failure_cases.csv')
    csv(exp[['scope','model_id','date','expID','n','constant_prediction','constant_target','true_tied_pairs','pred_tied_pairs','true_boundary_tie_size','prediction_boundary_tie_size']],r/'ties_and_coverage_audit.csv')
    csv(patch[(patch.y_true>=0)&(patch.y_true<=2)],r/'low_PCE_patch_audit.csv')
    # Label magnitude <1 affects only the auxiliary pair metric; main samples are unchanged.
    csv(exp[['scope','model_id','date','expID','pair_concordance','large_delta_ge1_concordance','small_delta_lt1_pair_count']],r/'small_PCE_difference_sensitivity.csv')
    csv(exp.groupby(['scope','model_id']).PCE_mae_pp.mean().reset_index(),r/'historical_PCE_MAE_existing_only.csv')
    reg=patch[patch.score_is_PCE_prediction].copy(); reg['abs_error']=abs(reg.score-reg.y_true)
    csv(reg.groupby(['scope','model_id','date']).abs_error.mean().reset_index().rename(columns={'abs_error':'PCE_MAE_pp'}),r/'historical_PCE_MAE_by_date.csv')
    records=summary.astype(object).where(summary.notna(),None).to_dict('records')
    psummary=ps.astype(object).where(ps.notna(),None).to_dict('records')
    result={'status':'ENGINEERING_COMPLETE','seed':SEED,'protocol_sha256':sha(BASE/'configs/frozen_p4_protocol.json'),
        'D2_rows':len(f),'labeled_rows':int(f.PCE_backward.notna().sum()),'evaluated_rows':len(evaluated),'eligible_experiments':len(eligible_keys),
        'excluded_insufficient_label_experiments':int((~eligible.eligible).sum()),'new_model_count':1,'top_quartile_rounding':'ceil','summary':records,'paired_summary':psummary,
        'P3_posthoc_reconciled':True,'naive_choices':choices,'all_patch_optical_inference':True,'holdout_used_for_selection':False,
        'wall_seconds':round(time.perf_counter()-start,3),'cpu_seconds':round(time.process_time()-cpu,3)}
    dump(r/'numeric_summary.json',result)
    dump(out/'run_receipt.json',{'argv':sys.argv,'status':'SUCCESS','python':sys.version,'wall_seconds':result['wall_seconds'],'cpu_seconds':result['cpu_seconds'],'random_replicates':c['random']['replicates'],'threads':1})
    print(json.dumps({'status':'SUCCESS','wall_seconds':result['wall_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); run(args.output)
