"""Bounded W1/W2 inner searches, then once-frozen nested outer evaluation."""
import argparse,copy,json,time,sys,gc
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import p7_core as p

def context_map(data):
    contexts={}
    for ds,d in data.items():
        allgroups=set(d['y'].split_group)
        for fold in sorted(d['folds'].outer_fold.unique()):
            ev=set(d['folds'].loc[d['folds'].outer_fold==fold,'split_group']); contexts[f'{ds}_OUTER_{int(fold)}']={'dataset':ds,'outer_fold':int(fold),'training_groups':sorted(allgroups-ev),'outer_validation_groups':sorted(ev)}
        contexts[ds+'_ALL']={'dataset':ds,'outer_fold':-1,'training_groups':sorted(allgroups),'outer_validation_groups':sorted(set(d['testopt'].split_group))}
    return contexts

def scope(ds,history=False): return ds+('_HISTORY_PREVIOUSLY_INSPECTED' if history else '_DEVELOPMENT_REUSED_NESTED_OOF')
def features(data): return {ds:{'dev':p.p6.d1_features(d['opt']) if ds=='D1' else d['opt'].copy(),'history':p.p6.d1_features(d['testopt']) if ds=='D1' else d['testopt'].copy()} for ds,d in data.items()}
def select_rows(frame,groups): return frame[frame.split_group.isin(set(groups))].reset_index(drop=True)

def fit_once(ds,vid,mod,y,f,ev,tag,resources,group_losses,fit_audit,negative=False):
    if set(y.split_group)&set(ev.split_group): raise ValueError('Fit/evaluation group leakage')
    start=time.perf_counter(); cpu=time.process_time()
    try: model=p.DVR(ds,p.view(vid),mod,p.config()).fit(y,f,negative=negative)
    except p.NotStable as exc:
        resources.append({**tag,'dataset':ds,'view_id':vid,'module_id':mod['module_id'],'status':'NOT_STABLE','reason':str(exc),'fit_wall_seconds':time.perf_counter()-start,'fit_cpu_seconds':time.process_time()-cpu,'infer_wall_seconds':np.nan,'infer_cpu_seconds':np.nan}); return None,'NOT_STABLE',None
    fw=time.perf_counter()-start; fc=time.process_time()-cpu; start=time.perf_counter(); cpu=time.process_time(); yp=model.predict(ev); iw=time.perf_counter()-start; ic=time.process_time()-cpu
    if not np.isfinite(yp).all(): raise ValueError('Nonfinite predicted score')
    resources.append({**tag,'dataset':ds,'view_id':vid,'module_id':mod['module_id'],'status':'SUCCESS','reason':'','n_features':model.feature_count_,'parameters':model.parameter_count_,'fit_wall_seconds':fw,'fit_cpu_seconds':fc,'infer_wall_seconds':iw,'infer_cpu_seconds':ic,'process_peak_MiB_to_date':p.p6.peak_memory_mib(),'per_model_memory_peak':'NOT_MEASURED'})
    audit={**tag,'dataset':ds,'view_id':vid,'module_id':mod['module_id'],'pair_hash':model.pair_hash_,'n_original_ordered_pairs':model.pair_count_,'training_pair_weight_mass':model.training_pair_mass_,'augmentation_copies':model.copies_,'objective':model.objective_,'gradient_inf':model.gradient_inf_,'iterations':model.iterations_,'solver_status':model.solver_status_,'training_groups':';'.join(sorted(model.training_groups_)),'evaluation_groups':';'.join(sorted(ev.split_group.unique())),'negative_shuffled_training_groups':';'.join(sorted(model.shuffled_groups_))}
    audit['training_optical_ID_sha256']=__import__('hashlib').sha256('\n'.join(sorted(f.sample_id)).encode()).hexdigest(); audit['training_scaler_sha256']=__import__('hashlib').sha256(model.pre.scaler_.mean_.tobytes()+model.pre.scaler_.scale_.tobytes()+model.pre.floor_.tobytes()).hexdigest()
    fit_audit.append(audit); diagnostic=model.group_diagnostic_.copy()
    for k,v in tag.items(): diagnostic[k]=v
    diagnostic['dataset']=ds; diagnostic['view_id']=vid; diagnostic['module_id']=mod['module_id']; group_losses.append(diagnostic)
    score=ev[p.KEYS].copy(); score['score']=yp
    return score,'SUCCESS',model

def metric_evaluate(y,sc,ds,mid,scope_name,fold):
    ids=set(y.sample_id); sc=sc[sc.sample_id.isin(ids)]
    if sc.score.isna().any():
        if not sc.score.isna().all() or set(sc.sample_id)!=ids: raise ValueError('Unexplained partial nonfinite score coverage')
        # Numerical failure is a method failure, never an ordering fallback or sample deletion.
        rows=[]
        for unit,g in y.groupby('unit_id',sort=True):
            rows.append({'dataset':ds,'scope':scope_name,'model_id':mid,'outer_fold':fold,'unit_id':unit,'split_group':str(g.split_group.iloc[0]),'n':len(g),'top_k':p.p6.top_k(len(g)),'status':'NOT_STABLE',**{m:np.nan for m in p.METRICS},'top25_true_pce_mean':np.nan,'experiment_true_pce_mean':g.target.mean()})
        return pd.DataFrame(rows)
    z=p.grouped_metrics(y,sc,ds,mid,scope_name,fold); z['status']='SUCCESS'; return z

def complete_method(group_metrics):
    return 'status' not in group_metrics or group_metrics.status.fillna('SUCCESS').eq('SUCCESS').all()

def m0(data,feature,c,out,resources):
    pc=p.p6.config(); rows=[]; reconc=[]; predictions=[]
    for ds,d in data.items():
        olddir=p.ROOT/c['P6_directory']/'results'; old=pd.read_csv(olddir/('d1_train_oof_predictions.csv' if ds=='D1' else 'd2_dev_oof_predictions.csv')).set_index('sample_id')
        oldmetrics=pd.read_csv(olddir/('d1_scores_by_substrate.csv' if ds=='D1' else 'd2_scores_by_experiment_and_date.csv')); oldmetrics=oldmetrics[oldmetrics.scope==pc[ds]['scope_dev']]
        mids=['PAIR_ABS','PL_PAIRWISE','ID_PLUS_PL_PAIRWISE','ID_PAIRWISE','LGBM_PL','EXTRATREES_PL']; specs={v['model_id']:v for v in p.p6.grid() if v['dataset']==ds}
        for fold in sorted(d['folds'].outer_fold.unique()):
            vals=set(d['folds'].loc[d['folds'].outer_fold==fold,'split_group']); train=set(d['y'].split_group)-vals; y=select_rows(d['y'],train); f=select_rows(feature[ds]['dev'],train); ev=select_rows(feature[ds]['dev'],vals); ey=select_rows(d['y'],vals)
            for mid in mids:
                spec=copy.deepcopy(specs.get(mid,{'dataset':ds,'model_id':mid,'family':'PAIRWISE','position':'ID' if mid=='ID_PAIRWISE' else 'IDPL' if mid=='ID_PLUS_PL_PAIRWISE' else 'PL','representation':'REL','statistics_subset':'ALL','window':'FULL'}))
                st=time.perf_counter(); ct=time.process_time(); model=p.p6.Model(ds,spec,pc).fit(y,f); fw=time.perf_counter()-st; fc=time.process_time()-ct; st=time.perf_counter(); pred=model.predict(ev); ip=time.perf_counter()-st
                sc=ev[p.KEYS].copy(); sc['score']=pred; sc=sc[sc.sample_id.isin(ey.sample_id)]; maxdiff=float(abs(sc.score.to_numpy()-old.loc[sc.sample_id,mid].to_numpy()).max())
                if maxdiff>=1e-8: raise ValueError(f'M0 score mismatch {ds} {mid}: {maxdiff}')
                mt=metric_evaluate(ey,sc,ds,'P6_'+mid,scope(ds),int(fold)); ref=oldmetrics[(oldmetrics.model_id==mid)&(oldmetrics.outer_fold==fold)].set_index('unit_id'); diff=max(float(abs(mt.set_index('unit_id')[m]-ref[m]).max()) for m in p.p6.METRICS)
                if diff>=1e-8: raise ValueError(f'M0 metric mismatch {ds} {mid}: {diff}')
                reconc.append({'dataset':ds,'outer_fold':int(fold),'method':mid,'max_abs_score_difference':maxdiff,'max_abs_metric_difference':diff,'status':'MATCH_WITHIN_1e-8'})
                sc['dataset']=ds; sc['model_id']='P6_'+mid; sc['outer_fold']=int(fold); predictions.append(sc); rows.append(mt); resources.append({'wave':'W1_M0','dataset':ds,'outer_fold':int(fold),'model_id':mid,'status':'SUCCESS','fit_wall_seconds':fw,'fit_cpu_seconds':fc,'infer_wall_seconds':ip,'process_peak_MiB_to_date':p.p6.peak_memory_mib()}); del model
            print(f'M0 {ds} outer={fold}: six strong baselines reproduced',flush=True)
        # P6 auxiliary and all remaining competitors retain their original frozen score tables.
        for mid in [s for s in old.columns if s in ['AUTHOR_KNN_AUX','ID_PRIOR','PL_INTENSITY','HISTGB_PL','HISTGB_IDPL','EXTRATREES_IDPL','LGBM_IDPL','RIDGE_PL','RIDGE_IDPL','F2_RIDGE_AUX']]:
            sc=d['y'][p.KEYS].copy(); sc['score']=old.loc[sc.sample_id,mid].to_numpy(); sc['outer_fold']=d['folds'].set_index('sample_id').loc[sc.sample_id,'outer_fold'].to_numpy(); sc['dataset']=ds; sc['model_id']='P6_'+mid; predictions.append(sc)
            for fold,g in sc.groupby('outer_fold'):
                ey=d['y'][d['y'].sample_id.isin(g.sample_id)]; rows.append(metric_evaluate(ey,g[p.KEYS+['score']],ds,'P6_'+mid,scope(ds),int(fold)))
    p.csv(pd.concat(rows,ignore_index=True),out/'M0_group_metrics.csv'); p.csv(pd.DataFrame(reconc),out/'M0_reproduction_checks.csv'); p.csv(pd.concat(predictions,ignore_index=True),out/'M0_dev_scores_frozen.csv')

def inner_wave(wave,data,feature,contexts,out,resources,group_losses,fit_audit,w1summary=None):
    allmetrics=[]; summary=[]; split_audit=[]
    for cid,ctx in contexts.items():
        ds=ctx['dataset']; y=select_rows(data[ds]['y'],ctx['training_groups']); f=select_rows(feature[ds]['dev'],ctx['training_groups']); chosen=None
        if wave=='W2': chosen=p.select_view(w1summary[w1summary.context_id==cid])
        candidates=p.views() if wave=='W1' else p.modules(); records={}; failed=set()
        for inner,tr,va in p.inner_splits(y):
            yy=select_rows(y,tr); ff=select_rows(f,tr); ev=select_rows(f,va); ey=select_rows(y,va)
            for role,gg in [('train',tr),('inner_validation',va)]:
                for group in sorted(gg): split_audit.append({'wave':wave,'context_id':cid,'dataset':ds,'outer_fold':ctx['outer_fold'],'inner_fold':inner,'role':role,'split_group':group})
            for spec in candidates:
                name=spec['view_id' if wave=='W1' else 'module_id']; vid=name if wave=='W1' else chosen; mod=p.module('BASE') if wave=='W1' else spec; tag={'wave':wave,'context_id':cid,'outer_fold':ctx['outer_fold'],'inner_fold':inner,'candidate_id':name}
                sc,status,model=fit_once(ds,vid,mod,yy,ff,ev,tag,resources,group_losses,fit_audit)
                if status!='SUCCESS': failed.add(name); continue
                mt=metric_evaluate(ey,sc,ds,name,'INNER_TRAIN_ONLY',inner); mt['context_id']=cid; mt['wave']=wave; mt['inner_fold']=inner; allmetrics.append(mt); records.setdefault(name,[]).append(mt); del model
            print(f'{wave} {cid} inner={inner}: {len(candidates)} candidates checked',flush=True)
        for spec in candidates:
            name=spec['view_id' if wave=='W1' else 'module_id']; row={'dataset':ds,'context_id':cid,'outer_fold':ctx['outer_fold'],'wave':wave,'view_id':name if wave=='W1' else chosen,'module_id':'BASE' if wave=='W1' else name,'status':'NOT_STABLE' if name in failed or len(records.get(name,[]))!=3 else 'SUCCESS'}
            if records.get(name):
                mt=pd.concat(records[name],ignore_index=True); mm=p.macro(mt,ds).iloc[0]
                for metric in p.METRICS: row[metric]=mm[metric]
            else:
                for metric in p.METRICS: row[metric]=np.nan
            vv=p.view(row['view_id']); row['n_features']=32*int(vv['position'])+12*int(vv['absolute'])+24*int(vv['relative']); summary.append(row)
        # Context checkpoints retain all unsuccessful and successful candidates.
        ck=out.parent/'checkpoints'/wave/cid; ck.mkdir(parents=True,exist_ok=True); p.csv(pd.DataFrame([r for r in summary if r['context_id']==cid]),ck/'candidate_summary.csv')
    p.csv(pd.concat(allmetrics,ignore_index=True),out/(wave+'_inner_group_metrics.csv')); p.csv(pd.DataFrame(summary),out/(wave+'_candidate_summary.csv')); p.csv(pd.DataFrame(split_audit),out/(wave+'_inner_split_audit.csv'))
    # Canonical saved precision is also used for selection, so checkpoint resume matches cold replay.
    return pd.read_csv(out/(wave+'_candidate_summary.csv'))

def choose_all(w1,w2,contexts,out):
    selection={}; ranking=[]
    for cid,ctx in contexts.items():
        ds=ctx['dataset']; other='D2' if ds=='D1' else 'D1'; othercid=other+'_ALL'; a=w1[w1.context_id==cid]; b=w1[w1.context_id==othercid]
        base=float(a.loc[a.view_id=='ABS','top25_gain_pp'].iloc[0]); obase=float(b.loc[b.view_id=='ABS','top25_gain_pp'].iloc[0]); mid,r=p.select_module(w2[w2.context_id==cid],w2[w2.context_id==othercid],base,obase); r['context_id']=cid; r['other_dataset_context']=othercid; ranking.append(r)
        selection[cid]={'dataset':ds,'outer_fold':ctx['outer_fold'],'view_id':p.select_view(a),'module_id':mid,'training_groups':ctx['training_groups'],'excluded_own_validation_groups':ctx['outer_validation_groups'],'other_dataset_evidence':othercid,'own_baseline_gain':base,'other_baseline_gain':obase,'used_own_outer_validation_labels':False,'used_history_labels':False}
    p.dump(out/'W3_frozen_selections.json',selection); p.csv(pd.concat(ranking,ignore_index=True),out/'joint_maximin_module_selection.csv'); return selection

def ablation_specs(sel):
    vid=sel['view_id']; chosen=p.module(sel['module_id']); base=p.module('BASE'); spec=[('W1_SELECTED',vid,base,False),('DVR_SELECTED',vid,chosen,False)]
    spec += [(name,v,base,False) for name,v in [('ABS_ERM','ABS'),('REL_ERM','REL'),('DUAL_ERM','DUAL'),('POS_DUAL_ERM','POS_DUAL')]]
    top=copy.deepcopy(base); top.update({'module_id':'FORCED_TOPK_ONLY','topk_boost':chosen['topk_boost'] if chosen['topk_boost']>1 else 2.}); robust=copy.deepcopy(base); robust.update({'module_id':'FORCED_ROBUST_ONLY','cvar_eta':chosen['cvar_eta'] if chosen['cvar_eta'] else .25,'cvar_q':chosen['cvar_q'] if chosen['cvar_eta'] else .25})
    spec += [('TOPK_ONLY',vid,top,False),('ROBUST_ONLY',vid,robust,False),('AUG_ONLY',vid,p.module('AUG'),False),('LIST_ONLY',vid,p.module('LIST10'),False)]
    for name,param,value in [('NO_TOPK','topk_boost',1.),('NO_ROBUST','cvar_eta',0.),('NO_AUGMENT','amplitude_augmentation',False),('NO_LISTWISE','listwise_lambda',0.)]:
        mod=copy.deepcopy(chosen); mod[param]=value; mod['module_id']=name; spec.append((name,vid,mod,False))
    spec.append(('NEGCTRL_SHUFFLED_HALF','DUAL',base,True)); return spec

def evaluate_w3(data,feature,contexts,selection,out,resources,group_losses,fit_audit):
    scoreparts=[]; metrics=[]; coefs=[]
    for history in [False,True]:
        if history: p.dump(out/'history_final_protocol_freeze.json',{'protocol_sha256':p.sha(p.BASE/'configs/frozen_p7_protocol.json'),'W3_selections_sha256':p.sha(out/'W3_frozen_selections.json'),'history_status':'PREVIOUSLY_INSPECTED','no_history_tuning':True})
        for cid,ctx in contexts.items():
            if (ctx['outer_fold']==-1)!=history: continue
            ds=ctx['dataset']; d=data[ds]; y=select_rows(d['y'],ctx['training_groups']); f=select_rows(feature[ds]['dev'],ctx['training_groups']); ev=feature[ds]['history'] if history else select_rows(feature[ds]['dev'],ctx['outer_validation_groups']); ey=d['testy_SCORING_ONLY'] if history else select_rows(d['y'],ctx['outer_validation_groups']); fold=ctx['outer_fold']; scparts=[]; fitted={}
            for name,vid,mod,negative in ablation_specs(selection[cid]):
                # Identical no-op ablations reuse an identical fitted model, not extra tuning.
                key=json.dumps({'view':vid,'module':{k:v for k,v in mod.items() if k!='module_id'},'negative':negative},sort_keys=True)
                if key in fitted:
                    sc,model=fitted[key]
                else:
                    tag={'wave':'W3','context_id':cid,'outer_fold':fold,'inner_fold':-1,'candidate_id':name}; sc,status,model=fit_once(ds,vid,mod,y,f,ev,tag,resources,group_losses,fit_audit,negative)
                    if status!='SUCCESS':
                        sc=ev[p.KEYS].copy(); sc['score']=np.nan
                        print(f'W3 {cid} {name}: NOT_STABLE; all rows kept, no fallback ranking',flush=True)
                    fitted[key]=(sc,model)
                scores=sc[sc.sample_id.isin(ey.sample_id)].copy(); scores['dataset']=ds; scores['model_id']=name; scores['scope']=scope(ds,history); scores['outer_fold']=fold; scparts.append(scores)
                if model is not None:
                    for j,n in enumerate(model.pre.names_): coefs.append({'dataset':ds,'scope':scope(ds,history),'outer_fold':fold,'model_id':name,'view_id':vid,'selected_module':selection[cid]['module_id'],'objective_module_name':mod['module_id'],'feature':n,'coefficient':model.coef_[j],'training_scaler_mean':model.pre.scaler_.mean_[j],'training_scaler_scale':model.pre.scaler_.scale_[j],'post_scaler_gate':model.pre.gate_[j],'training_relative_floor':np.nan if n.startswith('POS_') else model.pre.floor_[p.stats(ds).index(n.split('_',1)[1])]})
            frozen=pd.concat(scparts,ignore_index=True); file=out/(cid+'_scores_before_label_merge.csv'); p.csv(frozen,file)
            # The scoring-only target join follows the on-disk score freeze.
            for name,sc in frozen.groupby('model_id',sort=True): metrics.append(metric_evaluate(ey,sc[p.KEYS+['score']],ds,name,scope(ds,history),fold))
            scoreparts.append(frozen); print(f'W3 {cid}: two final endpoints + fixed ablations/negative control complete',flush=True); del fitted; gc.collect()
    pred=pd.concat(scoreparts,ignore_index=True); mt=pd.concat(metrics,ignore_index=True); p.csv(pred,out/'W3_all_scores_frozen.csv'); p.csv(mt,out/'W3_all_group_metrics.csv'); p.csv(pd.DataFrame(coefs),out/'W3_coefficients_and_preprocessing.csv')
    return pred,mt

def baseline_tables(data,out,c):
    frames=[]
    for ds,d in data.items():
        folder=p.ROOT/c['P6_directory']/'results'; old=pd.read_csv(folder/('d1_scores_by_substrate.csv' if ds=='D1' else 'd2_scores_by_experiment_and_date.csv'),dtype={'split_group':str}); old['model_id']='P6_'+old.model_id
        pc=p.p6.config(); old['scope']=old.scope.map({pc[ds]['scope_dev']:scope(ds),pc[ds]['scope_test']:scope(ds,True)})
        oracles={}
        for yy in [d['y'],d['testy_SCORING_ONLY']]:
            for unit,g in yy.groupby('unit_id'):
                vals=g.target.to_numpy(); k=p.p6.top_k(len(g)); oracle=vals[p.p6.order(vals,g.sample_id)[:k]].mean(); oracles[unit]=(oracle,vals.mean())
        old['top25_regret_pp']=[oracles[u][0]-(t if np.isfinite(t) else oracles[u][1]) for u,t in zip(old.unit_id,old.top25_true_pce_mean)]; frames.append(old)
    z=pd.concat(frames,ignore_index=True); p.csv(z,out/'P6_all_frozen_baseline_metrics_with_regret.csv'); return z

def final_statistics(mt,baseline,out,c):
    full=pd.concat([mt,baseline],ignore_index=True); p.csv(full,out/'all_methods_by_unit.csv'); macros=[]; units=[]; absolute=[]; deltas=[]; paired=[]
    for ds,q in full.groupby('dataset'):
        u=p.unit_metrics(q,ds); units.append(u)
        for (s,mid),g in u.groupby(['scope','model_id']):
            raw=q[(q.scope==s)&(q.model_id==mid)]; complete=complete_method(raw)
            row={'dataset':ds,'scope':s,'model_id':mid,'n_independent_groups':len(g),'rho_defined':int(g.rho.notna().sum()),'coverage_status':'COMPLETE' if complete else 'NOT_STABLE_INCOMPLETE','failed_units':int(raw.status.fillna('SUCCESS').ne('SUCCESS').sum()) if 'status' in raw else 0}
            for metric in p.METRICS:
                vals=g[metric].dropna().to_numpy() if complete else np.array([]); row[metric]=vals.mean() if len(vals) else np.nan
                row[metric+'_ci95_low']=np.nan; row[metric+'_ci95_high']=np.nan
                if len(vals):
                    rng=np.random.default_rng(p.SEED); boot=rng.choice(vals,size=(c['bootstrap']['replicates'],len(vals)),replace=True).mean(1); row[metric+'_ci95_low'],row[metric+'_ci95_high']=np.quantile(boot,[.025,.975])
            macros.append(row)
        references=['P6_PAIR_ABS','P6_LGBM_PL','P6_EXTRATREES_PL','P6_ID_PLUS_PL_PAIRWISE','P6_ID_PAIRWISE','P6_AUTHOR_KNN_AUX' if ds=='D1' else 'P6_F2_RIDGE_AUX']
        comparisons=[(a,b) for a in mt.model_id.unique() for b in references]+[(a,'DVR_SELECTED') for a in mt.model_id.unique() if a not in ['DVR_SELECTED','W1_SELECTED']]+[('DVR_SELECTED','W1_SELECTED'),('TOPK_ONLY','W1_SELECTED'),('ROBUST_ONLY','W1_SELECTED'),('AUG_ONLY','W1_SELECTED'),('LIST_ONLY','W1_SELECTED'),('NEGCTRL_SHUFFLED_HALF','DUAL_ERM')]
        for s,g in u.groupby('scope'):
            for a,b in sorted(set(comparisons)):
                if not complete_method(q[(q.scope==s)&q.model_id.isin([a,b])]): continue
                aa=g[g.model_id==a].set_index('split_group'); bb=g[g.model_id==b].set_index('split_group')
                if not len(aa) or not len(bb) or a==b: continue
                for metric in p.METRICS:
                    diff=(aa[metric]-bb[metric]).dropna(); vals=diff.to_numpy()
                    if not len(vals): continue
                    rng=np.random.default_rng(p.SEED); boot=rng.choice(vals,size=(c['bootstrap']['replicates'],len(vals)),replace=True).mean(1)
                    paired.append({'dataset':ds,'scope':s,'method_a':a,'method_b':b,'metric':metric,'mean_delta':vals.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),'improved':int((vals< -1e-12).sum()) if metric=='top25_regret_pp' else int((vals>1e-12).sum()),'worsened':int((vals>1e-12).sum()) if metric=='top25_regret_pp' else int((vals< -1e-12).sum()),'tied':int((abs(vals)<=1e-12).sum()),'n_independent_groups':len(vals)})
                    for group,v in diff.items(): deltas.append({'dataset':ds,'scope':s,'method_a':a,'method_b':b,'metric':metric,'split_group':group,'delta':v})
    macro=pd.DataFrame(macros); p.csv(macro,out/'macro_model_comparison.csv'); p.csv(pd.concat(units,ignore_index=True),out/'independent_group_metrics.csv'); p.csv(pd.DataFrame(paired),out/'paired_comparison_summary.csv'); p.csv(pd.DataFrame(deltas),out/'paired_group_deltas.csv'); p.csv(full[(full.top25_gain_pp<0)|(full.rho<0)|full.rho.isna()],out/'all_negative_and_failure_groups.csv')
    p.csv(mt[mt.model_id=='NEGCTRL_SHUFFLED_HALF'],out/'negative_control_group_metrics.csv')
    return macro,pd.DataFrame(paired)

def run(output,through='W3',resume=False):
    output=p.safe_dir(output); out=output/'results'
    if output.exists() and any(output.iterdir()) and not resume: raise ValueError('Preserve existing attempt; use a new directory or explicit --resume')
    out.mkdir(parents=True,exist_ok=True); c=p.config(); start=time.perf_counter(); cpu=time.process_time(); resources=[]; group_losses=[]; fit_audit=[]
    if resume and (out/'checkpoint_fit_audit.csv').exists():
        resources=pd.read_csv(out/'checkpoint_runtime.csv').to_dict('records'); fit_audit=pd.read_csv(out/'checkpoint_fit_audit.csv').to_dict('records'); group_losses=[pd.read_csv(out/'checkpoint_group_losses.csv')]
    with threadpool_limits(limits=1):
        data={'D1':p.oldrun.read_d1(p.p6.config()),'D2':p.oldrun.read_d2(p.p6.config())}; feature=features(data); contexts=context_map(data); p.dump(out/'outer_training_contexts.json',contexts); p.oldrun.audits(data,p.p6.config(),out)
        p.csv(pd.concat([pd.DataFrame({'dataset':ds,'sample_id':d['y'].sample_id,'unit_id':d['y'].unit_id,'split_group':d['y'].split_group,'patchID':d['y'].patchID,'outer_fold':d['folds'].outer_fold,'zero_PCE':d['y'].target==0}) for ds,d in data.items()],ignore_index=True),out/'canonical_identity_and_folds.csv')
        elapsed={}
        for wave in ['W1','W2']:
            checkpoint=output/'checkpoints'/(wave+'_COMPLETE.json'); now=time.perf_counter()
            if resume and checkpoint.exists():
                summary=pd.read_csv(out/(wave+'_candidate_summary.csv')); elapsed[wave]=json.loads(checkpoint.read_text())['wall_seconds']
            else:
                if wave=='W1': m0(data,feature,c,out,resources)
                summary=inner_wave(wave,data,feature,contexts,out,resources,group_losses,fit_audit,w1 if wave=='W2' else None); elapsed[wave]=round(time.perf_counter()-now,3)
                checkpoint.parent.mkdir(parents=True,exist_ok=True); p.dump(checkpoint,{'wave':wave,'status':'COMPLETE','wall_seconds':elapsed[wave],'candidate_summary_sha256':p.sha(out/(wave+'_candidate_summary.csv')),'protocol_sha256':p.sha(p.BASE/'configs/frozen_p7_protocol.json')})
                p.csv(pd.DataFrame(resources),out/'checkpoint_runtime.csv'); p.csv(pd.DataFrame(fit_audit),out/'checkpoint_fit_audit.csv'); p.csv(pd.concat(group_losses,ignore_index=True),out/'checkpoint_group_losses.csv')
                if elapsed[wave]>c['budgets_seconds'][wave]: raise TimeoutError(wave+' budget exceeded; stop branch and preserve checkpoint')
            if wave=='W1': w1=summary
            else: w2=summary
            if through==wave:
                p.csv(pd.DataFrame(resources),out/(wave+'_runtime_cost.csv')); p.csv(pd.DataFrame(fit_audit),out/(wave+'_training_fit_audit.csv')); return
        selection=choose_all(w1,w2,contexts,out); p.dump(out/'W3_program_and_selection_freeze.json',{'protocol_sha256':p.sha(p.BASE/'configs/frozen_p7_protocol.json'),'selection_sha256':p.sha(out/'W3_frozen_selections.json'),'source_sha256':{n:p.sha(p.BASE/'src'/n) for n in ['p7_core.py','run_p7.py']},'outer_evaluation_not_used_for_selection':True,'history_not_used_for_selection':True,'jointly_developed_on_two_datasets':True})
        now=time.perf_counter(); pred,mt=evaluate_w3(data,feature,contexts,selection,out,resources,group_losses,fit_audit); elapsed['W3']=round(time.perf_counter()-now,3)
        if elapsed['W3']>c['budgets_seconds']['W3']: raise TimeoutError('W3 budget exceeded')
        baseline=baseline_tables(data,out,c); macro,paired=final_statistics(mt,baseline,out,c)
        for ds,d in data.items():
            for hist in [False,True]:
                scores=pred[(pred.dataset==ds)&(pred.scope==scope(ds,hist))]; yy=d['testy_SCORING_ONLY'] if hist else d['y']; wide=scores.pivot(index='sample_id',columns='model_id',values='score').reset_index(); wide=yy.merge(wide,on='sample_id',validate='one_to_one'); p.csv(wide,out/(ds.lower()+('_history_predictions.csv' if hist else '_nested_oof_predictions.csv')))
        p.csv(pd.DataFrame(fit_audit),out/'training_fit_audit.csv'); p.csv(pd.concat(group_losses,ignore_index=True),out/'training_group_losses_and_robust_weights.csv'); p.csv(pd.DataFrame(resources),out/'runtime_and_resource_cost.csv')
        stable={'status':'ENGINEERING_COMPLETE','protocol_sha256':p.sha(p.BASE/'configs/frozen_p7_protocol.json'),'joint_development':True,'history_tuned':False,'W1_candidates':8,'W2_candidates':12,'final_endpoints':['W1_SELECTED','DVR_SELECTED'],'macro':json.loads(macro.to_json(orient='records')),'paired':json.loads(paired.to_json(orient='records')),'selections':selection,'M4':'NOT_RUN_OPTIONAL'}; p.dump(out/'numeric_summary.json',stable)
        p.dump(output/'checkpoints/W3_COMPLETE.json',{'status':'COMPLETE','wave':'W3','wall_seconds':elapsed['W3'],'numeric_summary_sha256':p.sha(out/'numeric_summary.json')})
    p.dump(output/'run_receipt.json',{'status':'SUCCESS','argv':sys.argv,'wall_seconds':round(time.perf_counter()-start,3),'cpu_seconds':round(time.process_time()-cpu,3),'wave_wall_seconds':elapsed,'process_peak_MiB':p.p6.peak_memory_mib(),'threads':1,'actual_fits':len(resources)})
    print(json.dumps({'status':'SUCCESS','wave_wall_seconds':elapsed,'wall_seconds':round(time.perf_counter()-start,3)}),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--through',choices=['W1','W2','W3'],default='W3'); ap.add_argument('--resume',action='store_true'); a=ap.parse_args(); run(a.output,a.through,a.resume)
