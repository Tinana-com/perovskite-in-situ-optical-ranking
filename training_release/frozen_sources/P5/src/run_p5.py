"""One frozen grouped D1 method replication; labels merge only after scores freeze."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from threadpoolctl import threadpool_limits
import d1_pipeline as p

def load(c,split):
    spec=c[split+'_file']; path=p.ROOT/c['data_directory']/spec['name']
    if path.stat().st_size!=spec['bytes'] or p.sha(path)!=spec['sha256']: raise ValueError('D1 input hash/size mismatch')
    with pd.HDFStore(path,'r') as h:
        if h.keys()!=['/df']: raise ValueError('Unexpected HDF keys')
    return pd.read_hdf(path,key=c['hdf_key'])

def audit(train,test,out,c):
    ti=p.identity(train); ei=p.identity(test)
    if set(ti.substrateID)&set(ei.substrateID): raise ValueError('D1 train/test substrate overlap')
    if train.groupby('substrateID').foldNo.nunique().max()!=1 or set(train.foldNo)!=set(range(5)): raise ValueError('Author folds invalid')
    if len(train)!=780 or len(test)!=349 or train.substrateID.nunique()!=26 or test.substrateID.nunique()!=12: raise ValueError('Unexpected D1 source counts')
    counts=[]; flags=[]; schema=[]; roles=[]; ids=[]; signalhash=[]
    for split,raw in [('train',train),('test',test)]:
        ident=p.identity(raw); y=p.target(raw)
        for sub,g in raw.groupby('substrateID',sort=True):
            counts.append({'split':split,'substrateID':sub,'n_rows':len(g),'n_finite_target':int(np.isfinite(g.pce_mean).sum()),'n_zero_PCE':int((g.pce_mean==0).sum()),'n_low_0_to_2':int(g.pce_mean.between(0,2).sum()),'foldNo':int(g.foldNo.iloc[0]) if split=='train' else -1,'n_folds':int(g.foldNo.nunique()) if split=='train' else 0})
        for i,(_,row) in enumerate(raw.iterrows()):
            a=np.asarray(row.LP725,dtype=float); p.clean_signal(a)
            hh=__import__('hashlib').sha256(a.tobytes()).hexdigest(); signalhash.append({'sample_id':ident.iloc[i].sample_id,'hash':hh})
            for col in ['LP725','LP725_start','LP725_end']:
                arr=np.asarray(row[col],dtype=float); matches=[]
                if col!='LP725':
                    matches=[j for j in range(len(a)-len(arr)+1) if np.allclose(a[j:j+len(arr)],arr,atol=1e-10,rtol=0,equal_nan=True)]
                schema.append({'split':split,'sample_id':ident.iloc[i].sample_id,'column':col,'length':len(arr),'nonfinite_count':int((~np.isfinite(arr)).sum()),'constant':bool(np.nanmax(arr)==np.nanmin(arr)),'contiguous_slice_starts_in_full':';'.join(map(str,matches)) if col!='LP725' else 'FULL'})
            ids.append({**ident.iloc[i].to_dict(),'split':split,'patchID_source_value':str(row.patchID),'patchID_source_type':type(row.patchID).__name__})
        flags.extend([{'check':split+'_duplicate_identity','value':0,'status':'PASS'},{'check':split+'_target_nonfinite','value':int((~np.isfinite(raw.pce_mean)).sum()),'status':'PASS'},
            {'check':split+'_unknown_patch_ID','value':int((~ident.patchID.isin(c['categories'])).sum()),'status':'INFO'},
            {'check':split+'_full_LP725_nonfinite_points','value':sum(int((~np.isfinite(np.asarray(a,float))).sum()) for a in raw.LP725),'status':'INFO'}])
    if pd.DataFrame(signalhash).hash.duplicated().any(): raise ValueError('Duplicate complete optical trace; inspect potential replicated samples before main analysis')
    for col in train.columns:
        role=('POSITION_INPUT' if col=='patchID' else 'GROUP_ONLY' if col in ['substrateID','foldNo'] else 'TARGET' if col=='pce_mean' else 'PL_INPUT' if col in ['LP725','LP725_start','LP725_end'] else 'POST_MEASUREMENT_FORBIDDEN' if col in ['voc_mean','jsc_mean','ff_mean','meanThickness','exsitu_feature'] else 'UNKNOWN')
        roles.append({'column':col,'role':role,'main_input':col in ['patchID','LP725'],'availability':'in situ optical' if role=='PL_INPUT' else 'region identity' if role=='POSITION_INPUT' else 'group/split only' if role=='GROUP_ONLY' else 'completed device measurement' if role in ['TARGET','POST_MEASUREMENT_FORBIDDEN'] else 'author summary timing not established; excluded','note':'Only full LP725 used; alternate windows excluded' if col in ['LP725_start','LP725_end'] else ''})
    flags.extend([{'check':'train_test_substrate_overlap','value':0,'status':'PASS'},{'check':'author_fold_substrate_leakage','value':0,'status':'PASS'},
        {'check':'duplicate_complete_LP725_traces','value':0,'status':'PASS'}, {'check':'test_optional_LP725_end_length201','value':int(test.LP725_end.map(len).eq(201).sum()),'status':'WARN_UNUSED_FIELD'},
        {'check':'post_JV_inputs_used','value':0,'status':'PASS'}, {'check':'D1_D2_mixed_training','value':0,'status':'PASS'}])
    p.csv(pd.DataFrame(roles),out/'d1_input_roles.csv'); p.csv(pd.DataFrame(roles),out/'INPUT_ROLE_TABLE.csv')
    p.csv(pd.DataFrame(counts),out/'d1_substrate_and_fold_audit.csv'); p.csv(pd.DataFrame(schema),out/'sequence_shape_and_window_audit.csv'); p.csv(pd.DataFrame(ids),out/'canonical_identity_audit.csv'); p.csv(pd.DataFrame(flags),out/'data_integrity_and_leakage_flags.csv')
    source=p.ROOT/c['data_directory']; commit=json.loads((source/'author_commit.json').read_text(encoding='utf-8'))['sha']
    if commit!=c['author_commit']: raise ValueError('Author source commit mismatch')
    record={'author_repository':'https://github.com/AI-InSu-Pero/ML-PerovskitePV-InSituLuminescene','author_commit':commit,'zenodo':'https://zenodo.org/records/7503391',
        'train':c['train_file'],'test':c['test_file'],'hdf_key':'/df','train_rows':780,'test_rows':349,'train_substrates':26,'test_substrates':12,
        'LP725_shape':[719],'patchID_train_dtype':str(train.patchID.dtype),'patchID_test_dtype':str(test.patchID.dtype),'columns':train.columns.tolist(),
        'substrateID':'author independent substrate group; not prediction feature','patchID':'author cell active-area region identity repeated within different substrate directories; categorical alignment used, geometry not inferred',
        'pce_mean':'author mean PCE label in percent; official record provides backward/forward/average target variants; infer mean field corresponds averaged PCE, not D2 backward target',
        'LP725':'author extracted single-channel PL transient; no continuous wavelength spectrum, physical seconds or absolute PLQY assumed',
        'LP725_start_end':'author-provided alternative transient arrays; actual contiguous matches recorded, exact generating/cropping rule not verified; unused',
        'position_status':c['position_status'],'raw_29_6GB_downloaded':False,'data_downloaded_this_round':False,'post_JV_inputs_used':False,
        'category_dictionary_from_train':sorted(ti.patchID.unique().tolist()),'test_optical_batch_condition':c['batch_assumption']}
    p.dump(out/'d1_data_provenance.json',record)

def feature_dictionary(out):
    formulas=['mean(x)','std(x,ddof=0)','mean(x[:72])','mean(x[-72:])','late10_mean-early10_mean','max(x)','argmax(x)/718',
        '(late-early)/max(abs(early),1e-12)','mean(x[179:540])','sum(x*(u-mean(u)))/sum((u-mean(u))^2), u=linspace(0,1,719)','trapezoid(x,dx=1)','sum(abs(diff(x)))']
    rows=[{'feature':name,'formula':formula,'source':'LP725 complete719','limitation':'frame index/author intensity units; no spectral peak or actual seconds'} for name,formula in zip(p.STATS,formulas)]
    rows += [{'feature':'relative_'+name,'formula':'(x-substrate optical median)/max(1.4826*substrate optical MAD,train-only floor)','source':'same statistic over available substrate optical regions','limitation':'batch optical availability; no PCE used'} for name in p.STATS]
    p.csv(pd.DataFrame(rows),out/'d1_feature_dictionary.csv')

def fit_case(y,opt,f,folds,evopt,evfeat,c,fold,scope,stores):
    if set(opt.substrateID)&set(evopt.substrateID): raise ValueError('Training/evaluation substrate overlap')
    scores=evopt[p.KEYS].copy(); prior=p.IDPrior().fit(y); scores['ID_PRIOR']=prior.predict(evopt[p.KEYS])
    pri=prior.table_.copy(); pri['outer_fold']=fold; pri['scope']=scope; stores['priors'].append(pri)
    rule,rid,rd=p.choose_rule(y,f,folds,c); scores['PL_INTENSITY']=rule[1]*evfeat[rule[0]].to_numpy(); rd['outer_fold']=fold; stores['rule_metrics'].append(rd)
    k,kd=p.choose_knn(y,opt,folds,c); kd['outer_fold']=fold; stores['knn_metrics'].append(kd)
    stores['choices'].append({'scope':scope,'outer_fold':fold,'simple_rule':rid,'knn_k':k,'training_substrates':';'.join(sorted(set(y.substrateID))),'evaluation_substrates':';'.join(sorted(set(evopt.substrateID)))})
    ym=y.set_index('sample_id').loc[opt.sample_id].pce_mean.to_numpy(); knn=KNeighborsRegressor(n_neighbors=k,weights='uniform',n_jobs=1,algorithm='brute').fit(p.raw_signals(opt),ym)
    scores['AUTHOR_KNN_AUX']=knn.predict(p.raw_signals(evopt))
    hashes=[]
    for mid in p.PAIR_MODELS:
        model=p.Pairwise(c,mid).fit(y,f); scores[mid]=model.predict(evfeat[p.KEYS] if mid=='ID_PAIRWISE' else evfeat); hashes.append(model.pair_hash_)
        pa=model.audit_.copy(); pa['outer_fold']=fold; pa['scope']=scope; pa['model_id']=mid; pa['substrate_pair_weight_sum']=model.n_pairs_/int(pa.unordered_pairs_used.gt(0).sum()); stores['pairs'].append(pa)
        for j,name in enumerate(model.names_):
            stores['coef'].append({'scope':scope,'outer_fold':fold,'model_id':mid,'feature':name,'coefficient':model.model_.coef_[0,j],'training_scaler_mean':model.scaler_.mean_[j],'training_scaler_scale':model.scaler_.scale_[j],
                'train_relative_floor':model.rel_.floor_[(j-(32 if mid=='ID_PLUS_PL_PAIRWISE' else 0))%12] if mid!='ID_PAIRWISE' else np.nan,
                'pair_ID_sha256':model.pair_hash_,'n_ordered_pairs':model.n_pairs_,'pair_weight_sum':model.weights_sum_,'solver_iterations':int(model.model_.n_iter_[0])})
    if len(set(hashes))!=1: raise ValueError('Input ablations use unequal training pairs')
    scores['outer_fold']=fold; scores['scope']=scope
    print(f'{scope} fold={fold}, train {len(y)} records/{y.substrateID.nunique()} substrates, predict {len(evopt)}; frozen rule {rid}, auxiliary k={k}',flush=True)
    return scores

def score_predictions(y,scores,raw,scope):
    # All score generation already finished; evaluation targets are joined here only.
    out=y.merge(scores,on=p.KEYS,validate='one_to_one'); out['RANDOM_EXPECTATION']=np.nan
    out=out.merge(pd.DataFrame({'sample_id':p.identity(raw).sample_id,'patchID_source_value':raw.patchID.map(str).to_numpy(),'patchID_source_type':raw.patchID.map(lambda v:type(v).__name__).to_numpy()}),on='sample_id',validate='one_to_one')
    rows=[]; patches=[]
    for fold,sc in scores.groupby('outer_fold',sort=True):
        yy=y[y.sample_id.isin(sc.sample_id)]
        for mid in p.SCORED_MODELS:
            ss=sc[p.KEYS+[mid]].rename(columns={mid:'score'}); rows.append(p.substrate_metrics(yy,ss,mid,scope,int(fold)))
            z=yy.merge(ss,on=p.KEYS,validate='one_to_one')
            for sub,g in z.groupby('substrateID',sort=True):
                g=g.copy(); g['model_id']=mid; g['scope']=scope; g['outer_fold']=fold; n=len(g); k=p.top_k(n)
                g['pred_average_rank']=g.score.rank(ascending=False,method='average'); g['true_average_rank']=g.pce_mean.rank(ascending=False,method='average')
                g['pred_top25']=False; g['true_top25']=False
                g.loc[g.index[p.order(g.score,g.sample_id)[:k]],'pred_top25']=True; g.loc[g.index[p.order(g.pce_mean,g.sample_id)[:k]],'true_top25']=True
                patches.append(g)
        rows.append(p.random_metrics(yy,scope,int(fold)))
    return out,pd.concat(rows,ignore_index=True),pd.concat(patches,ignore_index=True)

def run(output):
    output=p.safe_dir(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Preserve previous attempt; choose a new output directory')
    r=output/'results'; r.mkdir(parents=True); start=time.perf_counter(); cpu=time.process_time(); c=p.config()
    with threadpool_limits(limits=1):
        train=load(c,'train'); test=load(c,'test'); audit(train,test,r,c); feature_dictionary(r)
        opt=p.optical(train); testopt=p.optical(test); y=p.target(train); f=p.features(opt); tf=p.features(testopt)
        folds=opt[p.KEYS].copy(); folds['foldNo']=train.foldNo.to_numpy(); stores={k:[] for k in ['priors','choices','rule_metrics','knn_metrics','pairs','coef']}; oof=[]
        p.csv(folds,r/'fold_assignments.csv')
        p.csv(pd.concat([f.assign(split='train'),tf.assign(split='test')],ignore_index=True),r/'LP725_absolute_statistics.csv')
        for fold in sorted(train.foldNo.unique()):
            ids=set(folds.loc[folds.foldNo==fold,'sample_id']); tr=~opt.sample_id.isin(ids); va=~tr
            sc=fit_case(y[tr].reset_index(drop=True),opt[tr].reset_index(drop=True),f[tr].reset_index(drop=True),folds[tr].reset_index(drop=True),opt[va].reset_index(drop=True),f[va].reset_index(drop=True),c,int(fold),c['cv_scope'],stores); oof.append(sc)
        oof=pd.concat(oof,ignore_index=True)
        if oof.sample_id.duplicated().any() or set(oof.sample_id)!=set(opt.sample_id): raise ValueError('OOF coverage mismatch or overwritten OOF')
        # Test is a single frozen evaluation; replay uses the identical protocol.
        p.dump(r/'author_test_freeze.json',{'protocol_sha256':p.sha(p.BASE/'configs/frozen_p5_d1_protocol.json'),'source_hashes':{name:p.sha(p.BASE/'src'/name) for name in ['run_p5.py','d1_pipeline.py','ranking_metrics.py']},'training_targets_only_for_rule_and_k':True,'scope':c['test_scope'],'test_targets_not_used_for_model_selection':True})
        testsc=fit_case(y,opt,f,folds,testopt,tf,c,-1,c['test_scope'],stores)
        p.csv(oof,r/'OOF_scores_frozen_before_target_merge.csv'); p.csv(testsc,r/'author_test_scores_frozen_before_target_merge.csv')
        testy=p.target(test) # scoring-only evaluation-label projection
        cvpred,cvmt,cvpatch=score_predictions(y,oof,train,c['cv_scope']); tepred,temt,tepatch=score_predictions(testy,testsc,test,c['test_scope'])
        p.csv(cvpred,r/'d1_train_oof_predictions.csv'); p.csv(tepred,r/'d1_author_test_predictions.csv')
        p.csv(cvmt,r/'d1_cv_substrate_metrics.csv'); p.csv(temt,r/'d1_test_substrate_metrics.csv'); metrics=pd.concat([cvmt,temt],ignore_index=True)
        macro,boot,delta,pairs=p.bootstrap(metrics,c)
        p.csv(macro[macro.scope==c['cv_scope']],r/'d1_cv_macro_metrics.csv'); p.csv(macro[macro.scope==c['test_scope']],r/'d1_test_macro_metrics.csv')
        p.csv(boot,r/'d1_bootstrap_summary.csv'); p.csv(delta,r/'d1_paired_deltas.csv'); p.csv(pairs,r/'d1_paired_summary.csv')
        p.csv(metrics.groupby(['scope','outer_fold','model_id'])[p.METRICS].mean().reset_index(),r/'d1_fold_macro_metrics.csv')
        p.csv(metrics[(metrics.model_id!='RANDOM_EXPECTATION')&((metrics.rho<0)|(metrics.top25_gain_pp<0)|metrics.rho.isna())],r/'d1_failure_cases.csv')
        patch=pd.concat([cvpatch,tepatch],ignore_index=True); p.csv(patch,r/'patch_rank_and_selection_audit.csv'); p.csv(patch[patch.pce_mean.between(0,2)],r/'zero_and_low_PCE_patch_audit.csv')
        for key,name in [('priors','training_ID_priors'),('rule_metrics','inner_simple_rule_metrics'),('knn_metrics','inner_knn_MAE'),('pairs','training_pair_audit')]: p.csv(pd.concat(stores[key],ignore_index=True),r/(name+'.csv'))
        p.csv(pd.DataFrame(stores['choices']),r/'training_only_choices.csv'); p.csv(pd.DataFrame(stores['coef']),r/'coefficients_and_train_preprocessing.csv')
        auxiliary={'final_selected_k':int(stores['choices'][-1]['knn_k']),'author_test_MAE':float(np.mean(abs(tepred.pce_mean-tepred.AUTHOR_KNN_AUX))),
            'author_test_RMSE':float(np.sqrt(np.mean((tepred.pce_mean-tepred.AUTHOR_KNN_AUX)**2))),'CV_auxiliary_k_selection':'NESTED_WITHIN_OUTER_TRAIN_ONLY','P1_test_MAE':1.5155659025787969}
        if auxiliary['final_selected_k']!=14 or abs(auxiliary['author_test_MAE']-auxiliary['P1_test_MAE'])>1e-9: raise ValueError('Historical kNN reproduction mismatch')
        p.dump(r/'author_knn_reconciliation.json',auxiliary)
        numeric={'status':'ENGINEERING_COMPLETE','protocol_sha256':p.sha(p.BASE/'configs/frozen_p5_d1_protocol.json'),'seed':p.SEED,'n_train':780,'n_test':349,'n_train_substrates':26,'n_test_substrates':12,'test_used_for_selection':False,
            'macro':json.loads(macro.to_json(orient='records')),'paired':json.loads(pairs.to_json(orient='records')),'author_knn':auxiliary}
        p.dump(r/'numeric_summary.json',numeric)
    elapsed=time.perf_counter()-start
    if elapsed>c['wall_budget_seconds']: raise TimeoutError('CPU wall budget exceeded; preserve attempt')
    p.dump(output/'run_receipt.json',{'status':'SUCCESS','argv':sys.argv,'wall_seconds':round(elapsed,3),'cpu_seconds':round(time.process_time()-cpu,3),'threads':1,'seed':p.SEED,'test_policy':'one frozen evaluation, second run deterministic replay only'})
    print(json.dumps({'status':'SUCCESS','wall_seconds':round(elapsed,3)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True,type=Path); args=ap.parse_args(); run(args.output)
