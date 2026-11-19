"""Two datasets remain separate; development selection precedes historical scoring."""
import argparse,json,time,sys,gc
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import p6_core as p

def read_d1(c):
    cfg=c['D1']; raw=[]
    for split in ['train','test']:
        sp=cfg[split+'_file']; path=p.ROOT/cfg['data_directory']/sp['name']
        if path.stat().st_size!=sp['bytes'] or p.sha(path)!=sp['sha256']: raise ValueError('D1 source checksum mismatch')
        raw.append(pd.read_hdf(path,'df'))
    tr,te=raw
    if set(tr.substrateID)&set(te.substrateID) or tr.groupby('substrateID').foldNo.nunique().max()!=1: raise ValueError('D1 substrate leakage')
    if (len(tr),len(te),tr.substrateID.nunique(),te.substrateID.nunique())!=(780,349,26,12) or set(tr.foldNo)!=set(range(5)): raise ValueError('D1 unexpected groups')
    def projection(raw):
        meta=p.d1_identity(raw); opt=meta.copy(); opt['LP725']=raw.LP725.to_numpy(); y=meta.copy(); y['target']=pd.to_numeric(raw.pce_mean).to_numpy(float)
        if not np.isfinite(y.target).all(): raise ValueError('Nonfinite D1 label')
        return opt,y
    opt,y=projection(tr); evopt,evy=projection(te); folds=opt[p.KEYS].copy(); folds['outer_fold']=tr.foldNo.to_numpy()
    return {'opt':opt,'y':y,'testopt':evopt,'testy_SCORING_ONLY':evy,'folds':folds,'source_rows':1129,'optical_rows':1129,'train_groups':26,'test_groups':12,
        'schema_columns':list(tr.columns),'train_raw':tr,'test_raw':te}

def read_d2(c):
    cfg=c['D2']; path=p.ROOT/cfg['data_path']; feature=p.ROOT/cfg['features_path']
    if path.stat().st_size!=cfg['data_bytes'] or p.sha(path)!=cfg['data_sha256'] or p.sha(feature)!=cfg['features_sha256']: raise ValueError('D2 source/cache checksum mismatch')
    f=pd.read_csv(feature); raw=pd.read_hdf(path,'df')
    if len(f)!=11136 or f.PCE_backward.notna().sum()!=5857: raise ValueError('D2 source count mismatch')
    if not np.array_equal(raw.date.to_numpy(int),f.date.to_numpy(int)) or not np.array_equal(raw.patchID.to_numpy(int),f.patchID.to_numpy(int)) or not np.array_equal(raw.expID.to_numpy(str),f.expID.to_numpy(str)) or not np.allclose(raw.PCE_backward,f.PCE_backward,equal_nan=True): raise ValueError('D2 raw/cache identity mismatch')
    schema=list(raw.columns); del raw; gc.collect(); p.p4.optical(f)
    if f.groupby('expID').date.nunique().max()!=1 or f.duplicated(['date','expID','patchID']).any(): raise ValueError('D2 group identity invalid')
    meta=pd.DataFrame({'sample_id':f.sample_id,'unit_id':f.date.astype(str)+'::'+f.expID,'split_group':f.date.astype(str),'patchID':f.patchID.astype(int)})
    opt=pd.concat([meta,f[p.p4.F2]],axis=1); y=meta.copy(); y['target']=f.PCE_backward.to_numpy()
    counts=y.groupby('unit_id').target.count(); eligible=set(counts[counts>=4].index); y=y[y.target.notna()&y.unit_id.isin(eligible)].reset_index(drop=True)
    trdates=set(map(str,cfg['development_dates'])); tedates=set(map(str,cfg['history_dates'])); tr=opt[opt.split_group.isin(trdates)].reset_index(drop=True); te=opt[opt.split_group.isin(tedates)].reset_index(drop=True)
    yt=y[y.split_group.isin(trdates)].reset_index(drop=True); ye=y[y.split_group.isin(tedates)].reset_index(drop=True)
    folds=yt[p.KEYS].copy(); lookup={str(d):i for i,dates in enumerate(cfg['outer_validation_dates']) for d in dates}; folds['outer_fold']=folds.split_group.map(lookup)
    if len(yt)!=4707 or len(ye)!=1150 or yt.unit_id.nunique()!=158 or ye.unit_id.nunique()!=36: raise ValueError('D2 candidate coverage mismatch')
    return {'opt':tr,'y':yt,'testopt':te,'testy_SCORING_ONLY':ye,'folds':folds,'source_rows':11136,'optical_rows':11136,'train_groups':14,'test_groups':3,'schema_columns':schema,'eligibility':counts}

def frozen_csv(rel):
    path=p.ROOT/rel
    inv=pd.read_csv(p.BASE/'RUN_RECEIPTS/initial_source_inventory.csv').set_index('path')
    if p.sha(path)!=inv.loc[rel,'sha256']: raise ValueError('Frozen comparison source changed')
    return pd.read_csv(path)

def frozen_scores(ds,data,c,phase):
    ids=data['y' if phase=='dev' else 'testy_SCORING_ONLY'][p.KEYS]; scope=c[ds]['scope_dev' if phase=='dev' else 'scope_test']
    if ds=='D1':
        path=c['D1']['baseline_source']+'/results/'+('d1_train_oof_predictions.csv' if phase=='dev' else 'd1_author_test_predictions.csv'); old=frozen_csv(path)
        out=[]
        for mid in ['ID_PRIOR','PL_INTENSITY','AUTHOR_KNN_AUX']:
            sc=ids.merge(old[['sample_id',mid,'outer_fold']],on='sample_id',validate='one_to_one').rename(columns={mid:'score'}); sc['model_id']=mid; sc['scope']=scope; out.append(sc)
        return pd.concat(out,ignore_index=True)
    old=frozen_csv(c['D2']['baseline_source']+'/results/patch_scores_and_ranks.csv')
    wanted='EXPLORATORY_REUSED_DEV_DATES' if phase=='dev' else 'PREVIOUSLY_INSPECTED_HOLDOUT'; old=old[old.scope==wanted]
    mapping={'ID_PRIOR_SHRINK5':'ID_PRIOR','ID_PAIRWISE':'ID_PAIRWISE','ID_PLUS_PL_PAIRWISE':'ID_PLUS_PL_PAIRWISE','P4_PL_ONLY':'PL_PAIRWISE','P3_F2_RIDGE':'F2_RIDGE_AUX'}; out=[]
    for source,dest in mapping.items():
        sc=ids.merge(old.loc[old.model_id==source,['sample_id','score','outer_fold']],on='sample_id',validate='one_to_one'); sc['model_id']=dest; sc['scope']=scope; out.append(sc)
    naive=frozen_csv('research/pl_public_data_pilot_04/results/patch_scores_and_ranks.csv'); naive=naive[(naive.scope==wanted)&(naive.model_id=='NAIVE_INNER_SELECTED')]
    sc=ids.merge(naive[['sample_id','score','outer_fold']],on='sample_id',validate='one_to_one'); sc['model_id']='PL_INTENSITY'; sc['scope']=scope; out.append(sc)
    return pd.concat(out,ignore_index=True)

def feature_sets(ds,data,c,costs):
    result={}
    for phase,key in [('dev','opt'),('test','testopt')]:
        for window in (['FULL','EARLY25','LATE25'] if ds=='D1' else ['FULL']):
            start=time.perf_counter(); cpu=time.process_time(); f=p.d1_features(data[key],window) if ds=='D1' else data[key].copy()
            result[phase,window]=f
            costs.append({'dataset':ds,'scope':phase,'outer_fold':-99,'model_id':'FEATURES_'+window,'stage':'shared_feature_extraction','n_features':len(f.columns)-len(p.KEYS),'fit_wall_seconds':time.perf_counter()-start,'fit_cpu_seconds':time.process_time()-cpu,'infer_wall_seconds':np.nan,'infer_cpu_seconds':np.nan,'process_peak_working_set_MiB_to_date':p.peak_memory_mib(),'per_model_memory_peak':'NOT_MEASURED'})
    return result

def fit_models(ds,data,features,c,phase,costs,pairs,queries,preprocess,assignment):
    cfg=c[ds]; scores=[]; specs=[s for s in p.grid() if s['dataset']==ds]; y=data['y']; opt=data['opt']; folds=data['folds']; scope=cfg['scope_dev' if phase=='dev' else 'scope_test']
    cases=[]
    if phase=='dev':
        for fold in sorted(folds.outer_fold.unique()):
            vals=set(folds.loc[folds.outer_fold==fold,'split_group']); train_groups=set(opt.split_group)-vals; cases.append((int(fold),train_groups,vals))
    else: cases=[(-1,set(opt.split_group),set(data['testopt'].split_group))]
    for fold,train_groups,eval_groups in cases:
        if train_groups&eval_groups: raise ValueError('Split-group training/evaluation leakage')
        yt=y[y.split_group.isin(train_groups)].reset_index(drop=True); evy=data['y' if phase=='dev' else 'testy_SCORING_ONLY']; eid=evy[evy.split_group.isin(eval_groups)][p.KEYS].copy()
        for role,groups in [('train',train_groups),('evaluation',eval_groups)]:
            for group in sorted(groups): assignment.append({'dataset':ds,'scope':scope,'outer_fold':fold,'split_group':group,'role':role})
        for spec in specs:
            window=spec['window']; f=features['dev',window]; tr=f[f.split_group.isin(train_groups)].reset_index(drop=True)
            ef=features['dev' if phase=='dev' else 'test',window]; ev=ef[ef.split_group.isin(eval_groups)].reset_index(drop=True)
            if set(tr.unit_id)&set(ev.unit_id): raise ValueError('Optical units overlap')
            start=time.perf_counter(); cpu=time.process_time(); model=p.Model(ds,spec,c).fit(yt,tr); fitwall=time.perf_counter()-start; fitcpu=time.process_time()-cpu
            start=time.perf_counter(); cpu=time.process_time(); yp=model.predict(ev); infwall=time.perf_counter()-start; infcpu=time.process_time()-cpu
            pred=ev[p.KEYS].copy(); pred['score']=yp; pred=eid.merge(pred,on=p.KEYS,validate='one_to_one')
            if len(pred)!=len(eid) or not np.isfinite(pred.score).all(): raise ValueError('Prediction coverage or finite score failure')
            pred['model_id']=spec['model_id']; pred['scope']=scope; pred['outer_fold']=fold; scores.append(pred)
            costs.append({'dataset':ds,'scope':scope,'outer_fold':fold,'model_id':spec['model_id'],'stage':'fit_and_predict','n_features':len(model.pre_.names_),'fit_wall_seconds':fitwall,'fit_cpu_seconds':fitcpu,'infer_wall_seconds':infwall,'infer_cpu_seconds':infcpu,'process_peak_working_set_MiB_to_date':p.peak_memory_mib(),'per_model_memory_peak':'NOT_MEASURED'})
            for j,name in enumerate(model.pre_.names_):
                preprocess.append({'dataset':ds,'scope':scope,'outer_fold':fold,'model_id':spec['model_id'],'feature':name,'training_scaler_mean':model.pre_.scaler_.mean_[j],'training_scaler_scale':model.pre_.scaler_.scale_[j],
                    'pair_ID_sha256':getattr(model,'pair_hash_','NOT_PAIRWISE'),'n_ordered_pairs':getattr(model,'n_pairs_',0),'solver_iterations':int(model.model_.n_iter_[0]) if spec['family']=='PAIRWISE' else -1})
            floors=pd.DataFrame({'feature':model.pre_.stats_,'train_only_relative_floor':model.pre_.floor_}); floors['dataset']=ds; floors['scope']=scope; floors['outer_fold']=fold; floors['model_id']=spec['model_id']; preprocess.extend(floors.to_dict('records'))
            if model.pair_audit_ is not None:
                audit=model.pair_audit_.copy(); audit['dataset']=ds; audit['scope']=scope; audit['outer_fold']=fold; audit['model_id']=spec['model_id']; audit['pair_ID_sha256']=model.pair_hash_; audit['n_ordered_pairs']=model.n_pairs_; audit['pair_weight_sum']=model.pair_weight_sum_; pairs.append(audit)
            if model.query_audit_ is not None:
                audit=model.query_audit_.copy(); audit['dataset']=ds; audit['scope']=scope; audit['outer_fold']=fold; audit['model_id']=spec['model_id']; queries.append(audit)
            del model; gc.collect()
        print(f'{ds} {phase} fold={fold}: {len(specs)} fixed model configurations complete',flush=True)
    return pd.concat(scores,ignore_index=True)

def reconcile(ds,scores,data,c,phase):
    if ds=='D2': return pd.DataFrame([{'dataset':ds,'phase':phase,'method':'P4.1_FROZEN_CONTROLS','max_abs_score_difference':0.,'status':'VERIFIED_SOURCE_HASH_AND_ID'}])
    name='d1_train_oof_predictions.csv' if phase=='dev' else 'd1_author_test_predictions.csv'; old=frozen_csv(c['D1']['baseline_source']+'/results/'+name).set_index('sample_id'); rows=[]
    for mid in ['ID_PAIRWISE','PL_PAIRWISE','ID_PLUS_PL_PAIRWISE']:
        sc=scores[scores.model_id==mid]; diff=sc.score.to_numpy()-old.loc[sc.sample_id,mid].to_numpy(); maximum=float(abs(diff).max())
        rows.append({'dataset':ds,'phase':phase,'method':mid,'max_abs_score_difference':maximum,'status':'MATCH_WITHIN_1e-8' if maximum<1e-8 else 'MISMATCH'})
        if maximum>=1e-8: raise ValueError('Original P5 pairwise reproduction mismatch; preserve attempt')
    return pd.DataFrame(rows)

def prediction_table(scores,y,ds):
    wide=scores.pivot(index='sample_id',columns='model_id',values='score').reset_index(); meta=y.copy()
    if ds=='D1': meta['substrateID']=meta.unit_id; meta['pce_mean']=meta.target
    else: meta['date']=meta.split_group.astype(int); meta['expID']=meta.unit_id.str.split('::').str[-1]; meta['PCE_backward']=meta.target
    annotation=scores[['sample_id','scope','outer_fold']].drop_duplicates()
    if annotation.sample_id.duplicated().any(): raise ValueError('OOF annotation duplicated')
    z=meta.merge(wide,on='sample_id',validate='one_to_one').merge(annotation,on='sample_id',validate='one_to_one'); z['RANDOM_EXPECTATION']=np.nan
    if len(z)!=len(y): raise ValueError('Wide prediction candidates incomplete')
    return z

def audits(data,c,out):
    rows=[]; roles=[]
    for ds,dd in data.items():
        for side,key in [('development','y'),('history_PREVIOUSLY_INSPECTED','testy_SCORING_ONLY')]:
            y=dd[key]
            for split,g in y.groupby('split_group',sort=True): rows.append({'dataset':ds,'phase':side,'split_group':split,'n_labeled':len(g),'n_units':g.unit_id.nunique(),'n_zero_target':int((g.target==0).sum()),'n_low_0_to_2':int(g.target.between(0,2).sum()),'status':'PASS','duplicate_key':0,'finite_target':True})
        for col in dd['schema_columns']:
            optical=col=='LP725' if ds=='D1' else col in p.p4.CHANNELS
            roles.append({'dataset':ds,'source_column':col,'role':'PL_INPUT' if optical else 'POSITION_INPUT' if col=='patchID' else 'TARGET_SCORING_ONLY' if col in ['pce_mean','PCE_backward'] else 'GROUP_ONLY' if col in ['date','expID','substrateID','foldNo'] else 'FORBIDDEN_OR_UNUSED','is_direct_input':optical or col=='patchID'})
    p.csv(pd.DataFrame(rows),out/'input_and_group_audit.csv'); p.csv(pd.DataFrame(roles),out/'input_role_table.csv')

def run(output):
    output=p.safe_dir(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Existing attempt preserved; select a new path')
    r=output/'results'; r.mkdir(parents=True); c=p.config(); start=time.perf_counter(); cpu=time.process_time(); costs=[]; pairs=[]; queries=[]; preprocess=[]; assignment=[]; reconc=[]
    with threadpool_limits(limits=1):
        data={'D1':read_d1(c),'D2':read_d2(c)}; audits(data,c,r)
        feature={ds:feature_sets(ds,d,c,costs) for ds,d in data.items()}; results={}; dev_macro={}; dev_metrics={}; dev_units={}; selection={}
        # No historical evaluation occurs before all development candidates and selection freeze.
        for ds,d in data.items():
            sc=pd.concat([fit_models(ds,d,feature[ds],c,'dev',costs,pairs,queries,preprocess,assignment),frozen_scores(ds,d,c,'dev')],ignore_index=True)
            for mid,q in sc.groupby('model_id'):
                if len(q)!=len(d['y']) or set(q.sample_id)!=set(d['y'].sample_id) or q.sample_id.duplicated().any(): raise ValueError('OOF coverage incomplete')
            p.csv(sc,r/(ds.lower()+'_dev_scores_before_label_merge.csv'))
            reconc.append(reconcile(ds,sc,d,c,'dev')); mt=p.score_groups(d['y'],sc,ds); fm={str(g):int(k) for g,k in d['folds'][['split_group','outer_fold']].drop_duplicates().to_numpy()}; mt=pd.concat([mt,p.random_groups(d['y'],c[ds]['scope_dev'],ds,fm)],ignore_index=True)
            macro,units=p.summarize(mt,ds,c); dev_macro[ds]=macro; dev_metrics[ds]=mt; dev_units[ds]=units
            mains={s['model_id'] for s in p.grid() if s['dataset']==ds and s['role']=='MAIN'}|{'ID_PAIRWISE','PL_PAIRWISE','ID_PLUS_PL_PAIRWISE','ID_PRIOR','PL_INTENSITY','RANDOM_EXPECTATION','AUTHOR_KNN_AUX' if ds=='D1' else 'F2_RIDGE_AUX'}
            ranking=macro[macro.model_id.isin(mains)].sort_values(p.METRICS+['model_id'],ascending=[False,False,False,True],na_position='last'); winner=ranking.iloc[0].model_id
            selection[ds]={'rule':c['selection'],'selected_development_model':winner,'candidate_ids':ranking.model_id.tolist(),'historical_scores_used':False,'all_candidates_reported':True,'not_scientific_final_algorithm_selection':True}
            results[ds]={'dev_scores':sc,'dev_predictions':prediction_table(sc,d['y'],ds)}
        p.dump(r/'model_selection_train_only.json',selection); p.dump(r/'MODEL_SELECTION_TRAIN_ONLY.json',selection)
        p.dump(r/'history_evaluation_freeze.json',{'protocol_sha256':p.sha(p.BASE/'configs/frozen_p6_protocol.json'),'source_hashes':{name:p.sha(p.BASE/'src'/name) for name in ['p6_core.py','run_p6.py','ranking_metrics.py','p4_reference.py','p5_reference.py']},'development_selection_sha256':p.sha(r/'model_selection_train_only.json'),'history_status':'PREVIOUSLY_INSPECTED','no_history_model_selection':True})
        for ds,d in data.items():
            sc=pd.concat([fit_models(ds,d,feature[ds],c,'test',costs,pairs,queries,preprocess,assignment),frozen_scores(ds,d,c,'test')],ignore_index=True)
            p.csv(sc,r/(ds.lower()+'_history_scores_before_label_merge.csv'))
            reconc.append(reconcile(ds,sc,d,c,'test'))
            # Only scoring now joins held-out outcomes.
            y=d['testy_SCORING_ONLY']; mt=p.score_groups(y,sc,ds); fm={str(g):-1 for g in y.split_group.unique()}; mt=pd.concat([mt,p.random_groups(y,c[ds]['scope_test'],ds,fm)],ignore_index=True)
            macro,units=p.summarize(mt,ds,c); allmt=pd.concat([dev_metrics[ds],mt],ignore_index=True); allmacro=pd.concat([dev_macro[ds],macro],ignore_index=True); allunits=pd.concat([dev_units[ds],units],ignore_index=True)
            ids=sorted(set(allunits.model_id)); refs=['ID_PAIRWISE','PL_PAIRWISE','ID_PLUS_PL_PAIRWISE','AUTHOR_KNN_AUX' if ds=='D1' else 'F2_RIDGE_AUX']; contrasts=[(a,b) for a in ids for b in refs]
            ab=pd.read_csv(p.BASE/'configs/ablation_matrix.csv'); ab=ab[(ab.dataset==ds)&(ab.status=='RUN')]; contrasts += list(ab[['method_a','method_b']].itertuples(index=False,name=None))
            delta,paired=p.contrasts(allunits,contrasts,c)
            p.csv(delta,r/(ds.lower()+'_pairwise_method_deltas.csv')); p.csv(paired,r/(ds.lower()+'_paired_summary.csv'))
            p.csv(allmacro,r/(ds.lower()+'_macro_model_comparison.csv'))
            p.csv(allunits,r/(ds.lower()+'_resampling_unit_metrics.csv'))
            p.csv(allmt,r/('d1_scores_by_substrate.csv' if ds=='D1' else 'd2_scores_by_experiment_and_date.csv'))
            if ds=='D2': p.csv(allunits,r/'d2_scores_by_date.csv')
            legacy_roles={s['model_id']:s['role'] for s in p.grid() if s['dataset']==ds}
            abrows=ab.merge(paired.drop(columns='dataset'),on=['method_a','method_b'],how='left',validate='one_to_many')
            notrun=pd.read_csv(p.BASE/'configs/ablation_matrix.csv'); notrun=notrun[(notrun.dataset==ds)&(notrun.status=='NOT_RUN')]; abrows=pd.concat([abrows,notrun],ignore_index=True)
            p.csv(abrows,r/(ds.lower()+'_ablation_long.csv'))
            histpred=prediction_table(sc,y,ds); p.csv(results[ds]['dev_predictions'],r/('d1_train_oof_predictions.csv' if ds=='D1' else 'd2_dev_oof_predictions.csv')); p.csv(histpred,r/('d1_author_test_predictions.csv' if ds=='D1' else 'd2_history_predictions.csv'))
            results[ds].update({'macro':allmacro,'metrics':allmt,'paired':paired,'history_scores':sc})
        failures=pd.concat([q['metrics'] for q in results.values()],ignore_index=True); failures=failures[(failures.model_id!='RANDOM_EXPECTATION')&((failures.top25_gain_pp<0)|(failures.rho<0)|failures.rho.isna())]; p.csv(failures,r/'negative_and_failure_cases.csv')
        p.csv(pd.concat(reconc,ignore_index=True),r/'historical_baseline_reconciliation.csv'); p.csv(pd.concat(pairs,ignore_index=True),r/'training_pair_audit.csv'); p.csv(pd.concat(queries,ignore_index=True),r/'lambdarank_training_queries.csv'); p.csv(pd.DataFrame(preprocess),r/'training_preprocessing_parameters.csv'); p.csv(pd.DataFrame(assignment),r/'training_evaluation_group_assignments.csv')
        p.csv(pd.DataFrame(costs),r/'runtime_and_resource_cost.csv')
        numeric={'status':'ENGINEERING_COMPLETE','seed':p.SEED,'protocol_sha256':p.sha(p.BASE/'configs/frozen_p6_protocol.json'),'selection':selection,'models_new_fit':{'D1':17,'D2':8},'history_status':'PREVIOUSLY_INSPECTED','no_history_tuning':True,
            'macro':{ds:json.loads(q['macro'].to_json(orient='records')) for ds,q in results.items()},'input_sha256':{'D1_train':c['D1']['train_file']['sha256'],'D1_test':c['D1']['test_file']['sha256'],'D2':c['D2']['data_sha256']}}
        p.dump(r/'numeric_summary.json',numeric)
    elapsed=time.perf_counter()-start
    if elapsed>c['limits']['total_two_run_wall_budget_seconds']/2: raise TimeoutError('Single-run half-budget exceeded; preserve attempt')
    p.dump(output/'run_receipt.json',{'status':'SUCCESS','argv':sys.argv,'wall_seconds':round(elapsed,3),'cpu_seconds':round(time.process_time()-cpu,3),'peak_working_set_MiB_process':p.peak_memory_mib(),'threads':1,'models_fit_total':17*6+8*8})
    print(json.dumps({'status':'SUCCESS','wall_seconds':round(elapsed,3)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True,type=Path); a=ap.parse_args(); run(a.output)
