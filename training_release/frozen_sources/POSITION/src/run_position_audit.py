"""A single fixed position-confounding audit; all writes restricted to new scope."""
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
import p4_reference as p4
from ranking_metrics import group_metrics,aggregate,rho

KEYS=p4.KEYS; GROUP=p4.GROUP; F2=p4.F2; SEED=20261019
GRID=[[82,84,62,64,42,44,22,24],[81,83,61,63,41,43,21,23],[72,74,52,54,32,34,12,14],[71,73,51,53,31,33,11,13]]

def config(): return json.loads((BASE/'configs/frozen_position_protocol.json').read_text(encoding='utf-8'))
def sha(path): return p4.sha(path)
def dump(path,value): p4.dump(path,value)
def csv(frame,path): p4.csv(frame,path)
def safe_dir(path):
    path=Path(path).resolve()
    if path==BASE or not path.is_relative_to(BASE): raise ValueError('Writes restricted to position audit')
    return path

def id_frame(frame):
    if frame.sample_id.duplicated().any() or frame.duplicated(GROUP+['patchID']).any(): raise ValueError('Duplicate patch identity')
    if frame.groupby('expID').date.nunique().max()>1: raise ValueError('Experiment reused across dates')
    return frame[KEYS].copy()

def one_hot(frame,c):
    ids=pd.to_numeric(frame.patchID,errors='raise').to_numpy()
    cats=np.array(c['categorical_patchIDs'])
    if not np.isin(ids,cats).all(): raise ValueError('Unknown categorical patchID; physical layout must be audited separately')
    return (ids[:,None]==cats[None,:]).astype(float)

class IDPrior:
    def __init__(self,pseudocount=5.): self.pseudocount=pseudocount
    def fit(self,train):
        # Training experiment truth is permitted ONLY here, never in prediction.
        residual=train.PCE_backward-train.groupby(GROUP).PCE_backward.transform('mean')
        t=train[KEYS].copy(); t['training_residual']=residual.to_numpy()
        self.table_=t.groupby('patchID').training_residual.agg(['sum','count','mean']).reset_index()
        self.table_['prior']=self.table_['sum']/(self.table_['count']+self.pseudocount)
        self.lookup_=self.table_.set_index('patchID').prior
        return self
    def predict(self,ids):
        if set(ids.columns)!=set(KEYS): raise ValueError('ID-only prediction may not read labels or optics')
        return ids.patchID.map(self.lookup_).fillna(0.).to_numpy(float)

class ControlledPairwise:
    """Same loss/regularization for one-hot ID-only and ID+PL ablation."""
    def __init__(self,c,include_PL=True): self.c=c; self.include_PL=include_PL
    def fit(self,train_labels,train_opt):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        block=one_hot(train_opt,self.c)
        self.relative_=p4.Relative().fit(train_opt) if self.include_PL else None
        if self.include_PL: block=np.column_stack([block,self.relative_.transform(train_opt)])
        self.scaler_=StandardScaler().fit(block)
        lookup=pd.DataFrame(self.scaler_.transform(block),index=train_opt.sample_id)
        tr=train_labels.sort_values(['date','expID','sample_id']).reset_index(drop=True)
        x=lookup.loc[tr.sample_id].to_numpy()
        xx,yy,ww,self.pair_audit_,self.pair_ids_=p4.construct_pairs(tr,x,self.c['pairwise']['pair_tolerance_pp'])
        self.model_=LogisticRegression(random_state=SEED,**self.c['pairwise']['estimator'])
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            self.model_.fit(xx,yy,sample_weight=ww)
        self.n_pairs_=len(yy); self.weights_sum_=float(ww.sum())
        return self
    def predict(self,frame):
        allowed=KEYS+F2 if self.include_PL else KEYS
        if set(frame.columns)!=set(allowed): raise ValueError('Prediction API rejects evaluation targets/extra columns')
        block=one_hot(frame,self.c)
        if self.include_PL: block=np.column_stack([block,self.relative_.transform(frame)])
        return self.model_.decision_function(self.scaler_.transform(block))

def paired(date,c):
    rows=[]; summaries=[]
    for scope,ds in date.groupby('scope'):
        for a,b in c['primary_contrasts']+c['other_contrasts']:
            aa=ds[ds.model_id==a].set_index('date'); bb=ds[ds.model_id==b].set_index('date')
            for metric in ['rho','top25_gain_pp','top25_overlap']:
                delta=(aa[metric]-bb[metric]).dropna(); vals=delta.to_numpy(float)
                if not len(vals): continue
                rng=np.random.default_rng(SEED); boot=rng.choice(vals,size=(c['bootstrap']['replicates'],len(vals)),replace=True).mean(axis=1)
                for d,v in delta.items(): rows.append({'scope':scope,'method_a':a,'method_b':b,'metric':metric,'date':int(d),'delta':v,'seed':SEED})
                summaries.append({'scope':scope,'method_a':a,'method_b':b,'metric':metric,'mean_delta':vals.mean(),
                    'improved_dates':int((vals>1e-12).sum()),'worsened_dates':int((vals< -1e-12).sum()),'tied_dates':int((abs(vals)<=1e-12).sum()),
                    'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),'seed':SEED,'primary': [a,b] in c['primary_contrasts']})
    return pd.DataFrame(rows),pd.DataFrame(summaries)

def residual_diagnosis(pred):
    """Targets are scoring-only and constructed AFTER all predictions are fixed."""
    prior=pred[pred.model_id=='ID_PRIOR_SHRINK5'][['scope','sample_id','score']].rename(columns={'score':'training_ID_prior'})
    z=pred.merge(prior,on=['scope','sample_id'],validate='many_to_one')
    z['eval_experiment_mean_SCORING_ONLY']=z.groupby(['scope','model_id']+GROUP).y_true.transform('mean')
    z['within_exp_PCE_SCORING_ONLY']=z.y_true-z.eval_experiment_mean_SCORING_ONLY
    z['ID_adjusted_PCE_SCORING_ONLY']=z.within_exp_PCE_SCORING_ONLY-z.training_ID_prior
    groups=[]
    wanted=['P4_PL_ONLY','P3_F2_RIDGE','P3_RELATIVE_ET','P3_TWO_LEVEL','ID_PLUS_PL_PAIRWISE']
    for (scope,mid,date,exp),q in z[z.model_id.isin(wanted)].groupby(['scope','model_id']+GROUP):
        before=float(np.mean(q.within_exp_PCE_SCORING_ONLY**2)); after=float(np.mean(q.ID_adjusted_PCE_SCORING_ONLY**2))
        groups.append({'scope':scope,'model_id':mid,'date':int(date),'expID':exp,'n':len(q),
            'rho_score_vs_PCE_before_ID':rho(q.within_exp_PCE_SCORING_ONLY,q.score),
            'rho_score_vs_PCE_after_training_ID':rho(q.ID_adjusted_PCE_SCORING_ONLY,q.score),
            'rho_score_vs_training_ID_prior':rho(q.training_ID_prior,q.score),
            'within_exp_PCE_mse':before,'after_training_ID_mse':after,'ID_explained_fraction':1-after/before if before>0 else np.nan})
    return z[z.model_id.isin(wanted)],pd.DataFrame(groups)

def inventory(out,c):
    paths=[ROOT/c['data_path']]
    for key in ['p1_path','p2_path','p3_path','p4_path']:
        paths += [x for x in (ROOT/c[key]).rglob('*') if x.is_file() and not any(part.startswith(('.venv','.test_tmp','.mplconfig','__pycache__','.pytest_cache')) for part in x.parts)]
    csv(pd.DataFrame([{'path':str(x.relative_to(ROOT)),'bytes':x.stat().st_size,'sha256':sha(x),'access':'READ_ONLY','version':c['git_head']+' plus SHA256 for untracked artifacts'} for x in sorted(set(paths))]),out/'source_inventory.csv')

def position_evidence(out,c,f):
    image=BASE/'sources/Subtrate_Map.PNG'; tree=BASE/'sources/author_tree.json'
    verified=image.exists() and sha(image)=='4d619f189e470ef38afb592e52abe6483bced4e2df0075e0bdcf20f170373cc3'
    diagram=[{'patchID':v,'diagram_row_index':i,'diagram_column_index':j,'tens_group':v//10,'ones_local_ID':v%10,'not_a_model_input':True} for i,row in enumerate(GRID) for j,v in enumerate(row)]
    matches=sorted(f.patchID.unique().tolist())==sorted(row['patchID'] for row in diagram)
    evidence={'status':'AUTHOR_MAP_SPATIAL_ROI_LABELS_VERIFIED' if verified and matches else 'UNVERIFIED_POSITION_MEANING',
        'author_commit':'7ebe0c1ebff8b55d7d2ca14fe1ff05bff697fc43','diagram_sha256':sha(image) if image.exists() else None,
        'diagram_URL':'https://github.com/PerovskitePV/DL-Fabrication-Monitoring/blob/7ebe0c1ebff8b55d7d2ca14fe1ff05bff697fc43/images/Subtrate_Map.PNG',
        'dictionary_URL':'https://zenodo.org/records/14609789','dictionary_definition':'sample identifiers; cropped active-area temporal mean signals',
        'all_32_D2_IDs_match_author_map':matches,'interpretation':'Verified author diagram shows fixed image ROI locations, eight groups of four labeled regions; ID is not evidence of a simple row-major 8x4 grid',
        'unverified':['measurement order','physical length/spacing','camera orientation or mirror convention in every acquisition','whether each tens group is a distinct substrate from a textual data dictionary'],
        'coordinates_used_for_model':False}
    dump(out/'PATCH_ID_EVIDENCE.json',evidence); csv(pd.DataFrame(diagram),out/'author_map_transcription.csv')
    return evidence

def load_inputs(c):
    src=ROOT/c['data_path']
    if not src.is_file(): raise FileNotFoundError(f'Restore existing D2 at {src}; no synthetic fallback or download')
    if src.stat().st_size!=c['data_bytes'] or sha(src)!=c['data_sha256']: raise ValueError('D2 source checksum mismatch')
    if sha(ROOT/c['p4_path']/'PL_P4_RESULTS_FOR_CHATGPT.zip')!=c['p4_zip_sha256']: raise ValueError('P4 version mismatch')
    fp=ROOT/c['p2_path']/'results/extracted_features.csv'
    if f'{sha(fp)}  results/extracted_features.csv' not in (ROOT/c['p2_path']/'SHA256_MANIFEST.txt').read_text(encoding='utf-8'): raise ValueError('Feature cache checksum mismatch')
    f=pd.read_csv(fp); p4.optical(f)
    if len(f)!=11136 or f.PCE_backward.notna().sum()!=5857: raise ValueError('Unexpected real source counts')
    raw=pd.read_hdf(src,key='df')
    for col in ['date','patchID']:
        if not np.array_equal(pd.to_numeric(raw[col]).to_numpy(int),f[col].to_numpy(int)): raise ValueError('Raw identity mismatch')
    if not np.array_equal(raw.expID.astype(str),f.expID.astype(str)) or not np.allclose(raw.PCE_backward,f.PCE_backward,equal_nan=True,atol=1e-10): raise ValueError('Raw target/experiment mismatch')
    path=ROOT/c['p4_path']/'results/patch_scores_and_ranks.csv'
    if f'{sha(path)}  results/patch_scores_and_ranks.csv' not in (ROOT/c['p4_path']/'SHA256_MANIFEST.txt').read_text(encoding='utf-8'): raise ValueError('P4 predictions checksum mismatch')
    prior=pd.read_csv(path); prior=prior[prior.model_id.isin(c['frozen_PL_controls'])]
    return f,prior

def run(out):
    out=safe_dir(out)
    if out.exists() and any(out.iterdir()): raise ValueError('Existing attempt retained; choose a new directory')
    r=out/'results'; r.mkdir(parents=True); c=config(); start=time.perf_counter(); cpu=time.process_time()
    inventory(r,c); f,old=load_inputs(c); evidence=position_evidence(r,c,f)
    count=f.groupby(GROUP).agg(n_optical=('sample_id','size'),n_labeled=('PCE_backward','count')).reset_index(); count['eligible']=count.n_labeled>=4
    csv(count,r/'experiment_eligibility.csv')
    allowed=set(map(tuple,count.loc[count.eligible,GROUP].to_numpy()))
    labeled=f[f.PCE_backward.notna()&f[GROUP].apply(tuple,axis=1).isin(allowed)]
    pred=[]; priors=[]; coefficients=[]; paircounts=[]; folds=[]
    cases=[(i,[d for d in c['development_dates'] if d not in va],va,c['development_scope']) for i,va in enumerate(c['outer_validation_dates'])]
    cases.append((-1,c['development_dates'],c['holdout_dates'],c['holdout_scope']))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for fold,trdates,va,scope in cases:
            tr=f[f.date.isin(trdates)&f.PCE_backward.notna()]; ev=labeled[labeled.date.isin(va)]
            tr_opt=p4.optical(f[f.date.isin(trdates)]); ev_opt=p4.optical(f[f.date.isin(va)])
            if set(tr_opt.date)&set(ev_opt.date) or set(tr_opt.expID)&set(ev_opt.expID): raise ValueError('Date/experiment leakage')
            if fold==-1: dump(r/'holdout_freeze.json',{'protocol_sha256':sha(BASE/'configs/frozen_position_protocol.json'),'no_selection_or_retuning':True,'scope':scope})
            eid=id_frame(ev); all_ids=id_frame(ev_opt)
            for lam,mid in [(c['ID_prior']['shrink_pseudocount'],'ID_PRIOR_SHRINK5'),(0.,'ID_PRIOR_UNSHRUNK_DIAGNOSTIC')]:
                prior=IDPrior(lam).fit(tr); yp=prior.predict(eid)
                pred.append(p4.detail_predictions(ev,yp,mid,scope,fold))
                pt=prior.table_.copy(); pt['outer_fold']=fold; pt['scope']=scope; pt['pseudocount']=lam; pt['model_id']=mid; priors.append(pt)
            pred.append(p4.detail_predictions(ev,-ev.patchID.to_numpy(float),'ID_ASCENDING_DIAGNOSTIC',scope,fold))
            pred.append(p4.detail_predictions(ev,ev.patchID.to_numpy(float),'ID_DESCENDING_DIAGNOSTIC',scope,fold))
            for include,mid in [(False,'ID_PAIRWISE'),(True,'ID_PLUS_PL_PAIRWISE')]:
                model=ControlledPairwise(c,include).fit(tr,tr_opt)
                yp=pd.Series(model.predict(ev_opt if include else all_ids),index=ev_opt.sample_id).loc[ev.sample_id]
                pred.append(p4.detail_predictions(ev,yp,mid,scope,fold))
                names=[f'ID_{v}' for v in c['categorical_patchIDs']]+(F2 if include else [])
                phash=hashlib.sha256('\n'.join(model.pair_ids_).encode()).hexdigest()
                for j,name in enumerate(names): coefficients.append({'outer_fold':fold,'scope':scope,'model_id':mid,'feature':name,'coefficient':model.model_.coef_[0,j],
                    'scaler_mean_train_only':model.scaler_.mean_[j],'scaler_scale_train_only':model.scaler_.scale_[j],
                    'relative_floor_train_only':model.relative_.floor_[j-32] if include and j>=32 else np.nan,
                    'ordered_pair_count':model.n_pairs_,'pair_weight_sum':model.weights_sum_,'pair_IDs_sha256':phash,'solver_iterations':int(model.model_.n_iter_[0])})
                pa=model.pair_audit_.copy(); pa['outer_fold']=fold; pa['model_id']=mid; paircounts.append(pa)
            for src,dest in c['frozen_PL_controls'].items():
                q=old[(old.scope==scope)&(old.model_id==src)].set_index('sample_id').loc[ev.sample_id]
                if not np.allclose(q.y_true,ev.PCE_backward,atol=1e-10): raise ValueError('Frozen comparison samples/targets differ')
                pred.append(p4.detail_predictions(ev,q.score,dest,scope,fold,src!='PAIRWISE_LOGISTIC_REL12'))
            for d in trdates+va: folds.append({'outer_fold':fold,'scope':scope,'date':d,'role':'validation' if d in va else 'train'})
            print(f'{scope} fold={fold} train={len(tr)} eval={len(ev)} ID-only / ID+PL complete',flush=True)
            if time.process_time()-cpu>c['cpu_wall_budget_seconds']: raise TimeoutError('2-hour CPU budget exceeded; preserve attempt')
    predictions=pd.concat(pred,ignore_index=True); exp,patch=p4.score_groups(predictions); date,summary=aggregate(exp)
    csv(patch,r/'patch_scores_and_ranks.csv'); csv(exp,r/'metrics_by_experiment.csv'); csv(date,r/'metrics_by_date.csv'); csv(summary,r/'summary_date_macro.csv')
    delta,ps=paired(date,c); csv(delta,r/'paired_date_deltas.csv'); csv(ps,r/'paired_summary.csv')
    csv(pd.concat(priors),r/'training_ID_priors.csv'); csv(pd.DataFrame(coefficients),r/'coefficients_and_preprocessing.csv'); csv(pd.concat(paircounts),r/'training_pair_audit.csv'); csv(pd.DataFrame(folds),r/'fold_assignments.csv')
    residual,rd=residual_diagnosis(predictions); csv(residual,r/'ID_adjusted_residuals_SCORING_ONLY.csv'); csv(rd,r/'fixed_effect_diagnosis_by_experiment.csv')
    residual_date=rd.groupby(['scope','model_id','date']).agg(n_experiments=('expID','size'),rho_before_ID=('rho_score_vs_PCE_before_ID','mean'),rho_after_ID=('rho_score_vs_PCE_after_training_ID','mean'),rho_score_vs_ID=('rho_score_vs_training_ID_prior','mean'),PCE_mse_before=('within_exp_PCE_mse','mean'),PCE_mse_after=('after_training_ID_mse','mean')).reset_index()
    residual_date['ID_explained_fraction_from_date_MSE']=1-residual_date.PCE_mse_after/residual_date.PCE_mse_before
    csv(residual_date,r/'fixed_effect_diagnosis_by_date.csv')
    csv(exp[(exp.top25_gain_pp<0)|(exp.rho<0)|exp.rho.isna()],r/'all_failure_cases.csv')
    csv(patch[(patch.y_true>=0)&(patch.y_true<=2)],r/'low_PCE_patch_audit.csv')
    csv(exp[['scope','model_id','date','expID','n','constant_prediction','constant_target','true_tied_pairs','pred_tied_pairs','true_boundary_tie_size','prediction_boundary_tie_size']],r/'ties_and_coverage.csv')
    # No new model selection: empirical ID-prior quote is just reconciled retrospectively.
    recon=summary[summary.model_id.isin(['ID_PRIOR_UNSHRUNK_DIAGNOSTIC','ID_PRIOR_SHRINK5','P4_PL_ONLY','P3_F2_RIDGE'])].copy()
    csv(recon,r/'baseline_reconciliation.csv')
    records=summary.astype(object).where(summary.notna(),None).to_dict('records'); psummary=ps.astype(object).where(ps.notna(),None).to_dict('records')
    result={'status':'ENGINEERING_COMPLETE','seed':SEED,'protocol_sha256':sha(BASE/'configs/frozen_position_protocol.json'),
        'rows_total':len(f),'labeled_rows':int(f.PCE_backward.notna().sum()),'evaluated_rows':len(labeled),'eligible_experiments':int(count.eligible.sum()),
        'position_evidence':evidence,'model_classes':c['model_classes'],'summary':records,'paired_summary':psummary,'holdout_used_for_selection':False,
        'wall_seconds':round(time.perf_counter()-start,3),'cpu_seconds':round(time.process_time()-cpu,3)}
    dump(r/'numeric_summary.json',result)
    dump(out/'run_receipt.json',{'argv':sys.argv,'status':'SUCCESS','python':sys.version,'wall_seconds':result['wall_seconds'],'cpu_seconds':result['cpu_seconds'],'threads':1})
    print(json.dumps({'status':'SUCCESS','wall_seconds':result['wall_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True,type=Path); a=ap.parse_args(); run(a.output)
