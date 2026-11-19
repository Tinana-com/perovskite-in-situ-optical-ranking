"""Three-level independent-group stacking; one frozen selected P8 endpoint."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,copy,json,time,sys,gc
from pathlib import Path
import p8_core as p
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

def rows(frame,groups): return frame[frame.split_group.isin(set(groups))].reset_index(drop=True)
def scope(ds,history=False): return ds+('_HISTORY_PREVIOUSLY_INSPECTED' if history else '_DEVELOPMENT_REUSED_NESTED_OOF')
def contexts(data):
    result={}
    for ds,d in data.items():
        gg=set(d['y'].split_group)
        for fold in sorted(d['folds'].outer_fold.unique()):
            va=set(d['folds'].loc[d['folds'].outer_fold==fold,'split_group']); result[f'{ds}_OUTER_{int(fold)}']={'dataset':ds,'outer_fold':int(fold),'train':sorted(gg-va),'validation':sorted(va),'history':False}
        result[ds+'_ALL']={'dataset':ds,'outer_fold':-1,'train':sorted(gg),'validation':sorted(d['testopt'].split_group.unique()),'history':True}
    return result

class ScoreBank:
    """Only training labels held here; validation candidate IDs contain no outcome."""
    def __init__(self,ds,data,dev_features,history_features,output,start):
        self.ds=ds; self.y=data['y'].copy(deep=True); self.opt=pd.concat([dev_features,history_features],ignore_index=True); self.candidate_ids=pd.concat([data['y'][p.KEYS],data['testy_SCORING_ONLY'][p.KEYS]],ignore_index=True); self.output=output; self.start=start; self.cache={}; self.resources=[]; self.fit_audit=[]; self.uses=[]; self.crossfit_audit=[]
    def spec(self,name):
        mid={'BASE':'PAIR_ABS','POS':'ID_PAIRWISE','TREE':'EXTRATREES_PL'}[name]
        exact=next((s for s in p.p6.grid() if s['dataset']==self.ds and s['model_id']==mid),None)
        return copy.deepcopy(exact or {'dataset':self.ds,'model_id':mid,'family':'PAIRWISE','position':'ID','representation':'REL','window':'FULL','statistics_subset':'ALL'})
    def get(self,train,validation,role,forbidden=()):
        train=set(train); validation=set(validation); forbidden=set(forbidden)
        if train&validation or train&forbidden or validation&forbidden: raise ValueError('Base/aux cross-fit leakage')
        if not train<=set(self.y.split_group): raise ValueError('Historical labels forbidden in model fitting')
        key=self.ds+'_'+p.group_hash(['T:'+x for x in train]+['V:'+x for x in validation])[:24]
        self.uses.append({'dataset':self.ds,'role':role,'bank_id':key,'train_group_ids_hash':p.group_hash(train),'validation_group_ids_hash':p.group_hash(validation),'train_groups':';'.join(sorted(train)),'validation_groups':';'.join(sorted(validation)),'forbidden_groups':';'.join(sorted(forbidden)),'used_validation_targets':False})
        if key in self.cache: return self.cache[key].copy()
        if time.perf_counter()-self.start>p.config()['budgets']['per_run_wall_seconds']: raise TimeoutError('P8 per-run budget reached; retain checkpoints')
        yy=rows(self.y,train); ff=rows(self.opt,train); ev=rows(self.opt,validation); keys=rows(self.candidate_ids,validation); result=keys.copy(); specs=p.p6.config()
        for name in ['BASE','POS','TREE']:
            st=time.perf_counter(); cpu=time.process_time(); model=p.p6.Model(self.ds,self.spec(name),specs).fit(yy,ff); fw=time.perf_counter()-st; fc=time.process_time()-cpu; st=time.perf_counter(); cpu=time.process_time(); values=model.predict(ev); iw=time.perf_counter()-st; ic=time.process_time()-cpu
            if not np.isfinite(values).all(): raise ValueError('NOT_STABLE auxiliary/base; preserve affected attempt')
            score=ev[p.KEYS].copy(); col='base_score' if name=='BASE' else name+'_score'; score[col]=values; result=result.merge(score,on=p.KEYS,validate='one_to_one')
            self.resources.append({'dataset':self.ds,'bank_id':key,'model':name,'n_features':len(model.pre_.names_),'linear_parameters':int(model.model_.coef_.size) if name!='TREE' else np.nan,'tree_count':len(model.model_.estimators_) if name=='TREE' else 0,'tree_nodes':sum(e.tree_.node_count for e in model.model_.estimators_) if name=='TREE' else 0,'fit_wall_seconds':fw,'fit_cpu_seconds':fc,'infer_wall_seconds':iw,'infer_cpu_seconds':ic,'process_peak_MiB_to_date':p.p6.peak_memory_mib(),'per_model_peak_MiB':'NOT_MEASURED'})
            self.fit_audit.append({'dataset':self.ds,'bank_id':key,'model':name,'train_groups':';'.join(sorted(train)),'validation_groups':';'.join(sorted(validation)),'train_group_ids_hash':p.group_hash(train),'validation_group_ids_hash':p.group_hash(validation),'train_labeled_sample_ids_hash':p.id_hash(yy.sample_id),'train_optical_sample_ids_hash':p.id_hash(ff.sample_id),'validation_sample_ids_hash':p.id_hash(keys.sample_id),'scaler_sha256':__import__('hashlib').sha256(model.pre_.scaler_.mean_.tobytes()+model.pre_.scaler_.scale_.tobytes()+model.pre_.floor_.tobytes()).hexdigest(),'pair_hash':getattr(model,'pair_hash_','NOT_PAIRWISE'),'parameters':json.dumps(model.spec,sort_keys=True),'status':'SUCCESS'})
            del model; gc.collect()
        if len(result)!=len(keys): raise ValueError('Bank candidate row loss')
        folder=self.output/'crossfit_evidence'/key; folder.mkdir(parents=True,exist_ok=True); p.csv(result,folder/'unlabeled_scores.csv'); p.dump(folder/'identity.json',{'dataset':self.ds,'train_groups':sorted(train),'validation_groups':sorted(validation),'train_hash':p.group_hash(train),'validation_hash':p.group_hash(validation),'labels_in_score_file':False})
        self.cache[key]=result.copy(deep=True); return result
    def crossfit(self,groups,role,forbidden=()):
        yy=rows(self.y,groups); frames=[]
        for fold,tr,va in p.split_groups(yy):
            sc=self.get(tr,va,role+f'_META_CF_{fold}',forbidden); frames.append(sc)
            for g in sorted(va): self.crossfit_audit.append({'dataset':self.ds,'role':role,'meta_crossfit_fold':fold,'heldout_meta_group':g,'base_aux_training_groups':';'.join(sorted(tr)),'crossfit_scope_groups':';'.join(sorted(groups)),'scope_forbidden_groups':';'.join(sorted(forbidden)),'crossfit_is_out_of_sample':g not in tr,'current_gate_validation_labels_used':False})
        z=pd.concat(frames,ignore_index=True)
        if z.sample_id.duplicated().any() or set(z.sample_id)!=set(yy.sample_id): raise ValueError('Incomplete gate-training OOF')
        return z

def meta_training(bank,groups,role,forbidden,output,evidence):
    sc=bank.crossfit(groups,role,forbidden); yy=rows(bank.y,groups); optical=rows(bank.opt,groups); result={}
    for aux in ['POS','TREE']:
        ss=p.feature_scores(sc,aux); meta=p.gate_features(ss,optical,bank.ds); labels=p.training_supervision(yy,ss,groups); result[aux]=(meta,labels)
        z=meta.copy(); z['dataset']=bank.ds; z['context_role']=role; z['auxiliary']=aux; z['source']='BASE_AUX_TRAIN_SCOPE_OOF'; z['gate_training_scope_hash']=p.group_hash(groups); evidence['meta'].append(z)
        z=labels.copy(); z['dataset']=bank.ds; z['context_role']=role; z['auxiliary']=aux; evidence['supervision'].append(z)
    return result
def gate_for(spec,training,cache,role,evidence):
    if spec['kind']!='LEARNED': return None
    key=(spec['aux'],spec.get('drop_disagreement',False),spec.get('negative',False))
    if key not in cache:
        st=time.perf_counter(); cpu=time.process_time(); meta,y=training[spec['aux']]; gate=p.MetaGate(key[1],key[2]).fit(meta,y); cache[key]=gate
        evidence['gate_parameters'].append({'context_role':role,'auxiliary':spec['aux'],'drop_disagreement':key[1],'negative_control':key[2],**gate.parameters()})
        evidence['gate_cost'].append({'context_role':role,'auxiliary':spec['aux'],'drop_disagreement':key[1],'negative_control':key[2],'fit_wall_seconds':time.perf_counter()-st,'fit_cpu_seconds':time.process_time()-cpu,'n_training_meta_rows':len(meta),'n_independent_groups':meta.split_group.nunique(),'parameter_count':len(gate.names_)+1 if gate.model_ is not None else 0,'status':gate.status_})
    return cache[key]

def score_specs(ds,bank,raw,eval_keys,specs,training,role,evidence,output,fold,phase):
    if len({s['model_id'] for s in specs})!=len(specs): raise ValueError('Duplicate frozen prediction registry identifiers')
    cache={}; scoreframes=[]; annotations=[]; visible={}
    for aux in ['POS','TREE']:
        ss=p.feature_scores(raw,aux); meta=p.gate_features(ss,rows(bank.opt,set(raw.split_group)),ds); visible[aux]=(ss,meta)
        z=meta.copy(); z['dataset']=ds; z['context_role']=role; z['auxiliary']=aux; z['source']='UNLABELED_GATE_INFERENCE'; evidence['inference_meta'].append(z)
    for spec in specs:
        gate=gate_for(spec,training,cache,role,evidence); ss,meta=visible[spec['aux']]; pred,ann=p.infer_policy(ss,meta,spec,gate)
        pred['model_id']=spec['model_id']; pred['dataset']=ds; pred['scope']=phase; pred['outer_fold']=fold; pred['context_role']=role; scoreframes.append(pred)
        ann['model_id']=spec['model_id']; ann['dataset']=ds; ann['scope']=phase; ann['outer_fold']=fold; ann['context_role']=role; annotations.append(ann)
    scores=pd.concat(scoreframes,ignore_index=True); anns=pd.concat(annotations,ignore_index=True); checkpoint=output/'checkpoints'/role; checkpoint.mkdir(parents=True,exist_ok=True); p.csv(scores,checkpoint/'scores_before_label_merge.csv'); p.csv(anns,checkpoint/'lambda_before_label_merge.csv')
    # This function accepts candidate IDs only. True outcomes are joined by the caller after save.
    if not set(eval_keys.sample_id)==set(scores.sample_id) or scores.duplicated(['model_id','sample_id']).any(): raise ValueError('Inference row coverage changed')
    return scores,anns

def target_serialization_difference(values,reference):
    difference=float(np.max(abs(np.asarray(values,float)-np.asarray(reference,float))))
    if difference>1e-10: raise ValueError('Historical target values changed beyond frozen CSV precision')
    return difference

def reconcile(bank,raw,data,fold,history=False):
    ds=bank.ds; file=('d1_author_test_predictions.csv' if history else 'd1_train_oof_predictions.csv') if ds=='D1' else ('d2_history_predictions.csv' if history else 'd2_dev_oof_predictions.csv'); folder=p.ROOT/'research/pl_public_data_pilot_06_baselines_ablations/results'; ref=pd.read_csv(folder/file).set_index('sample_id'); yy=data['testy_SCORING_ONLY' if history else 'y']; yy=yy[yy.sample_id.isin(raw.sample_id)]
    if set(raw.sample_id)!=set(yy.sample_id): raise ValueError('W0 target identity mismatch')
    target_difference=target_serialization_difference(yy.target,ref.loc[yy.sample_id,'target'])
    rows_=[]
    for col,mid in [('base_score','PAIR_ABS'),('POS_score','ID_PAIRWISE'),('TREE_score','EXTRATREES_PL')]:
        maximum=float(abs(raw[col].to_numpy()-ref.loc[raw.sample_id,mid].to_numpy()).max()); equal=[]
        for _,g in raw.groupby('unit_id'): equal.append(p.selected(g[col],g.sample_id)==p.selected(ref.loc[g.sample_id,mid],g.sample_id))
        if maximum>=1e-8 or not all(equal): raise ValueError(f'W0 mismatch {ds} {mid}: {maximum}, Top25same={all(equal)}')
        rows_.append({'dataset':ds,'scope':scope(ds,history),'outer_fold':fold,'model_id':mid,'max_abs_score_difference':maximum,'max_abs_target_serialization_difference':target_difference,'all_Top25_sets_identical':all(equal),'n_samples':len(raw),'status':'MATCH_WITHIN_1e-8_AND_EXACT_SELECTED_IDS'})
    return rows_

def select_context(cid,ctx,bank,output,specs,evidence,selection_audit):
    T=set(ctx['train']); outer=set(ctx['validation']); y=rows(bank.y,T); metrics=[]
    for inner,U,V in p.split_groups(y):
        role=cid+f'_SELECT_{inner}'; raw=bank.get(U,V,role+'_VALIDATION',outer); training=meta_training(bank,U,role+'_TRAIN',outer|V,output,evidence)
        keys=rows(bank.candidate_ids,V); scores,_=score_specs(bank.ds,bank,raw,keys,specs,training,role,evidence,output,inner,'TRAIN_INTERNAL_SELECTION')
        truth=rows(bank.y,V)
        for mid,g in scores.groupby('model_id',sort=True): metrics.append(p.evaluate(truth,g,bank.ds,mid,'TRAIN_INTERNAL_SELECTION',inner))
        selection_audit.append({'context_id':cid,'dataset':bank.ds,'selection_fold':inner,'selection_training_groups':';'.join(sorted(U)),'selection_validation_groups':';'.join(sorted(V)),'outer_excluded_groups':';'.join(sorted(outer)),'train_group_ids_hash':p.group_hash(U),'validation_group_ids_hash':p.group_hash(V),'fresh_gate_OOF_scope':';'.join(sorted(U)),'meta_OOF_source_excludes_selection_validation':True,'used_outer_targets':False,'used_history_targets':False})
        print(role+': true nested base/aux OOF -> gate -> untouched selection groups complete',flush=True)
    mt=pd.concat(metrics,ignore_index=True); summary=p.candidate_summary(mt,specs); summary['context_id']=cid; summary['dataset']=bank.ds
    specmap={s['model_id']:s for s in specs}; families={'FIXED_SELECTED':[s['model_id'] for s in specs if s['wave']=='W1'],'RULE_SELECTED':[s['model_id'] for s in specs if s['kind']=='RULE'],'LEARNED_SELECTED':[s['model_id'] for s in specs if s['kind']=='LEARNED' and not s['cap']],'CAP_SELECTED':[s['model_id'] for s in specs if s['kind']=='LEARNED' and s['cap']]}
    selected={'P8_SELECTED':p.choose(summary),**{k:p.choose(summary,v) for k,v in families.items()}}
    folder=output/'checkpoints'/cid; folder.mkdir(parents=True,exist_ok=True); p.csv(summary,folder/'all_inner_candidates.csv'); p.csv(mt,folder/'all_inner_unit_metrics.csv'); p.dump(folder/'selected.json',{'choices':selected,'outer_or_history_targets_used':False,'training_groups':sorted(T),'excluded_groups':sorted(outer)})
    return selected,summary,mt

def frozen_comparators(data):
    old=pd.read_csv(p.ROOT/'research/pl_public_data_pilot_07_robust_ranking/results/P6_all_frozen_baseline_metrics_with_regret.csv',dtype={'split_group':str}); old=old[old.model_id!='P6_PAIR_ABS'].copy(); old['source']='P6_FROZEN_AUXILIARY'
    previous=pd.read_csv(p.ROOT/'research/pl_public_data_pilot_07_robust_ranking/results/W3_all_group_metrics.csv',dtype={'split_group':str}); previous=previous[previous.model_id=='DVR_SELECTED'].copy(); previous['model_id']='P7_DVR_PREVIOUS'; previous['source']='P7_FROZEN_PREVIOUSLY_ANALYZED'
    full=pd.concat([old,previous],ignore_index=True)
    for (ds,s,mid),q in full.groupby(['dataset','scope','model_id']):
        yy=data[ds]['testy_SCORING_ONLY' if 'HISTORY' in s else 'y']
        if set(q.unit_id)!=set(yy.unit_id): raise ValueError('Frozen baseline comparison unit mismatch')
    return full

def statistics(metrics,out):
    u=p.independent_metrics(metrics); p.csv(u,out/'independent_group_metrics.csv'); macro=[]; paired=[]; unit_deltas=[]; tails=[]; experimental=[]; c=p.config()
    for (ds,s),full in u.groupby(['dataset','scope'],sort=True):
        base=full[full.model_id=='P6_PAIR_ABS'].set_index('split_group')
        for mid,g in full.groupby('model_id',sort=True):
            q=g.set_index('split_group'); assert set(q.index)==set(base.index); row={'dataset':ds,'scope':s,'model_id':mid,'n_independent_groups':len(g),'rho_defined':int(g.rho.notna().sum())}
            for metric in p.METRICS:
                v=g[metric].dropna().to_numpy(); row[metric]=float(v.mean()) if len(v) else np.nan; row[metric+'_ci95_low']=np.nan; row[metric+'_ci95_high']=np.nan
                if len(v):
                    rng=np.random.default_rng(p.SEED); boot=rng.choice(v,size=(c['bootstrap']['replicates'],len(v)),replace=True).mean(1); row[metric+'_ci95_low'],row[metric+'_ci95_high']=np.quantile(boot,[.025,.975])
                delta=(q[metric]-base[metric]).dropna(); vals=delta.to_numpy()
                for group,value in delta.items(): unit_deltas.append({'dataset':ds,'scope':s,'model_id':mid,'reference':'P6_PAIR_ABS','metric':metric,'split_group':group,'delta':value,'candidate_value':q.loc[group,metric],'baseline_value':base.loc[group,metric]})
                if len(vals):
                    rng=np.random.default_rng(p.SEED); boot=rng.choice(vals,size=(c['bootstrap']['replicates'],len(vals)),replace=True).mean(1); paired.append({'dataset':ds,'scope':s,'model_id':mid,'reference':'P6_PAIR_ABS','metric':metric,'mean_delta':float(vals.mean()),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),'improved':int((vals>1e-10).sum()) if metric!='top25_regret_pp' else int((vals< -1e-10).sum()),'worsened':int((vals< -1e-10).sum()) if metric!='top25_regret_pp' else int((vals>1e-10).sum()),'tied':int((abs(vals)<=1e-10).sum()),'n_independent_groups':len(vals)})
            macro.append(row); diff=q.top25_gain_pp-base.top25_gain_pp; tail=p.tail_values(diff); tails.append({'dataset':ds,'scope':s,'model_id':mid,'reference':'P6_PAIR_ABS','mean_delta_pp':float(diff.mean()),'worst_group':str(diff.idxmin()),'minimum_absolute_gain_pp':float(q.top25_gain_pp.min()),'absolute_negative_gain_groups':int((q.top25_gain_pp<0).sum()),**tail})
    p.csv(pd.DataFrame(macro),out/'macro_model_comparison.csv'); p.csv(pd.DataFrame(paired),out/'paired_comparison_summary.csv'); p.csv(pd.DataFrame(unit_deltas),out/'per_unit_paired_deltas.csv'); p.csv(pd.DataFrame(tails),out/'failure_tail_audit.csv')
    # Experiment/substrate-level paired details complement the independent-date risk table.
    for (ds,s),g in metrics.groupby(['dataset','scope']):
        b=g[g.model_id=='P6_PAIR_ABS'][['unit_id','top25_gain_pp']].rename(columns={'top25_gain_pp':'baseline_gain_pp'})
        q=g.merge(b,on='unit_id',validate='many_to_one'); q['delta_gain_pp']=q.top25_gain_pp-q.baseline_gain_pp; experimental.append(q)
    p.csv(pd.concat(experimental,ignore_index=True),out/'all_unit_comparisons_and_failures.csv')
    p.csv(metrics[metrics.scope.str.contains('HISTORY')],out/'history_previously_inspected.csv'); p.csv(metrics[metrics.model_id.str.contains('NO_DISAGREE|NEGATIVE_META|RAW_FIXED|LEARNED|FIX_|RULE|ALONE')],out/'ablation_comparison.csv')
    return pd.DataFrame(macro),pd.DataFrame(tails)

def run(output):
    output=p.safe_dir(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Existing attempt preserved; use a new output directory')
    out=output/'results'; out.mkdir(parents=True); start=time.perf_counter(); cpu=time.process_time(); c=p.config(); specs=c['W1']+c['W2']; evidence={k:[] for k in ['meta','supervision','inference_meta','gate_parameters','gate_cost']}; selection_audit=[]; summaries=[]; inner_metrics=[]; all_scores=[]; all_annotations=[]; metrics=[]; reconc=[]; elapsed={}
    with threadpool_limits(limits=1):
        data={'D1':p.oldrun.read_d1(p.p6.config()),'D2':p.oldrun.read_d2(p.p6.config())}; p.oldrun.audits(data,p.p6.config(),out); feature={ds:{'dev':p.p6.d1_features(d['opt']) if ds=='D1' else d['opt'].copy(),'history':p.p6.d1_features(d['testopt']) if ds=='D1' else d['testopt'].copy()} for ds,d in data.items()}; ctxs=contexts(data); p.dump(out/'outer_training_contexts.json',ctxs)
        banks={ds:ScoreBank(ds,d,feature[ds]['dev'],feature[ds]['history'],output,start) for ds,d in data.items()}
        identity=pd.concat([d['folds'].assign(dataset=ds) for ds,d in data.items()],ignore_index=True); p.csv(identity,out/'canonical_identity_and_folds.csv')
        now=time.perf_counter()
        for cid,ctx in ctxs.items():
            if ctx['history']: continue
            ds=ctx['dataset']; raw=banks[ds].get(ctx['train'],ctx['validation'],cid+'_W0'); reconc.extend(reconcile(banks[ds],raw,data[ds],ctx['outer_fold'])); print('W0 '+cid+': ABS/POS/TREE scores and Top25 identities reproduced',flush=True)
        p.csv(pd.DataFrame(reconc),out/'W0_reproduction_checks.csv'); elapsed['W0']=time.perf_counter()-now
        # All hyperparameter choices are finalized before any new outer candidate is scored.
        choices={}; now=time.perf_counter()
        for cid,ctx in ctxs.items():
            chosen,summary,mt=select_context(cid,ctx,banks[ctx['dataset']],output,specs,evidence,selection_audit); choices[cid]={'choices':chosen,'dataset':ctx['dataset'],'training_groups':ctx['train'],'validation_groups':ctx['validation'],'used_outer_targets':False,'used_history_targets':False}; summaries.append(summary); mt['context_id']=cid; inner_metrics.append(mt)
        elapsed['W1_W2_true_nested_selection']=time.perf_counter()-now
        p.dump(out/'selected_protocol.json',choices); p.csv(pd.concat(summaries,ignore_index=True),out/'all_candidates.csv'); p.csv(pd.concat(inner_metrics,ignore_index=True),out/'all_inner_unit_metrics.csv'); p.csv(pd.DataFrame(selection_audit),out/'model_selection_and_leakage_audit.csv')
        p.dump(out/'W3_protocol_and_selection_freeze.json',{'protocol_sha256':p.sha(p.BASE/'configs/frozen_p8_protocol.json'),'selection_sha256':p.sha(out/'selected_protocol.json'),'main_source_sha256':{n:p.sha(p.BASE/'src'/n) for n in ['p8_core.py','run_p8.py']},'outer_candidate_outcomes_used':False,'history_outcomes_used':False,'final_endpoint':'P8_SELECTED'})
        now=time.perf_counter(); specmap={s['model_id']:s for s in specs}
        for cid,ctx in sorted(ctxs.items(),key=lambda item:(item[1]['history'],item[0])):
            ds=ctx['dataset']; bank=banks[ds]; T=set(ctx['train']); V=set(ctx['validation']); role=cid+'_FROZEN'; training=meta_training(bank,T,role+'_TRAIN',V,output,evidence); raw=bank.get(T,V,role+'_INFERENCE'); allspec=copy.deepcopy(specs)+p.diagnostic_specs(specs)
            for name,selected in choices[cid]['choices'].items():
                spec=copy.deepcopy(specmap[selected]); spec['model_id']=name; spec['eligible']=False; allspec.append(spec)
            yy=data[ds]['testy_SCORING_ONLY' if ctx['history'] else 'y']; yy=rows(yy,V); phase=scope(ds,ctx['history']); scores,ann=score_specs(ds,bank,raw,yy[p.KEYS],allspec,training,role,evidence,output,ctx['outer_fold'],phase)
            if ctx['history']: reconc.extend(reconcile(bank,raw,data[ds],ctx['outer_fold'],True))
            for mid,g in scores.groupby('model_id',sort=True): metrics.append(p.evaluate(yy,g,ds,mid,phase,ctx['outer_fold']))
            all_scores.append(scores); all_annotations.append(ann); print('W3 '+cid+': frozen inference saved before scoring; all diagnostics complete',flush=True)
        elapsed['W3_frozen_evaluation']=time.perf_counter()-now
        pred=pd.concat(all_scores,ignore_index=True); ann=pd.concat(all_annotations,ignore_index=True); mt=pd.concat(metrics,ignore_index=True); base=mt[mt.model_id=='NOOP'].copy(); base['model_id']='P6_PAIR_ABS'; base['source']='P8_REPRODUCED_P6_ABS'; mt=pd.concat([mt,base,frozen_comparators(data)],ignore_index=True); p.csv(mt,out/'all_methods_by_unit.csv'); p.csv(ann,out/'lambda_and_rank_change_by_group.csv'); p.csv(pred[pred.selected_top25],out/'top25_selected_sample_ids.csv')
        for ds,d in data.items():
            q=pred[pred.dataset==ds]; p.csv(q,out/(ds+'_unlabeled_predictions.csv')); truth=pd.concat([d['y'].assign(scope=scope(ds)),d['testy_SCORING_ONLY'].assign(scope=scope(ds,True))],ignore_index=True); truth['dataset']=ds; p.csv(truth,out/(ds+'_evaluation_labels_ONLY.csv'))
            joined=q.merge(truth,on=p.KEYS+['scope','dataset'],validate='many_to_one'); p.csv(joined,out/(ds+'_predictions_SCORING_ONLY.csv'))
        macro,tails=statistics(mt,out); p.csv(pd.DataFrame(reconc),out/'W0_reproduction_checks.csv')
        p.csv(pd.concat(evidence['meta']+evidence['inference_meta'],ignore_index=True),out/'group_level_gate_features.csv'); p.csv(pd.concat(evidence['supervision'],ignore_index=True),out/'gate_supervision_TRAIN_ONLY.csv'); p.dump(out/'gate_fitted_parameters.json',evidence['gate_parameters']); p.csv(pd.DataFrame(evidence['gate_cost']),out/'gate_runtime_cost.csv')
        p.csv(pd.DataFrame([x for b in banks.values() for x in b.fit_audit]),out/'base_aux_training_audit.csv'); p.csv(pd.DataFrame([x for b in banks.values() for x in b.uses]),out/'all_bank_uses_and_exclusions.csv'); p.csv(pd.DataFrame([x for b in banks.values() for x in b.crossfit_audit]),out/'gate_double_OOF_lineage.csv'); p.csv(pd.DataFrame([x for b in banks.values() for x in b.resources]),out/'base_aux_runtime_cost.csv')
        for ds in ['D1','D2']:
            q=mt[mt.dataset==ds]; p.csv(q,out/(ds+'_metrics_by_substrate.csv' if ds=='D1' else 'D2_metrics_by_experiment.csv'))
        p.dump(out/'numeric_summary.json',{'status':'ENGINEERING_COMPLETE','protocol_sha256':p.sha(p.BASE/'configs/frozen_p8_protocol.json'),'selected':choices,'final_endpoint':'P8_SELECTED','history_previously_inspected':True,'no_P9':True,'macro':json.loads(macro.to_json(orient='records')),'tails':json.loads(tails.to_json(orient='records'))})
    elapsed_total=time.perf_counter()-start
    if elapsed_total>c['budgets']['per_run_wall_seconds']: raise TimeoutError('P8 single run budget exceeded')
    p.dump(output/'run_receipt.json',{'status':'SUCCESS','argv':sys.argv,'wall_seconds':round(elapsed_total,3),'cpu_seconds':round(time.process_time()-cpu,3),'wave_wall_seconds':{k:round(v,3) for k,v in elapsed.items()},'optical_model_fits':sum(len(b.resources) for b in banks.values()),'meta_gate_fits':len(evidence['gate_cost']),'process_peak_MiB':p.p6.peak_memory_mib(),'threads':1,'no_new_packages_or_downloads':True})
    print(json.dumps({'status':'SUCCESS','wall_seconds':round(elapsed_total,3),'optical_fits':sum(len(b.resources) for b in banks.values())}),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); run(a.output)
