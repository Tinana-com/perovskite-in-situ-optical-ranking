"""Re-fit only frozen P6 M0 on development folds, audit raw optical traces."""
import argparse,time,json,sys,importlib.metadata as md,platform,gc
from pathlib import Path
import r0_core as r
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

def optical_audit(data,out):
    records=[]; schemas=[]; examples=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for row,(ds,d) in enumerate(data.items()):
        channels=['LP725'] if ds=='D1' else r.CHANNELS
        for col in d['schema']:
            schemas.append({'dataset':ds,'field':col,'role':'PL_OPTICAL_INPUT' if col in ['LP725','LP780','SP775'] else 'OPTICAL_CONTROL_NOT_PL' if col=='ND' else 'TRAIN_TARGET_OR_SCORING_ONLY' if col in ['pce_mean','PCE_backward'] else 'IDENTITY_ONLY' if col in ['substrateID','date','expID','patchID','foldNo'] else 'FORBIDDEN_OR_UNUSED'})
        # Complete optical-container scan, including identity-only historical units; no targets here.
        for q in d['raw_opt'].itertuples(index=False):
            for ch in channels:
                a=np.asarray(getattr(q,ch),float); z,s=r.resample_shape(a)
                records.append({'dataset':ds,'sample_id':q.sample_id,'split_group':q.split_group,'channel':ch,'length':len(a),'ndim':a.ndim,'dtype':str(a.dtype),'finite_count':int(np.isfinite(a).sum()),'nonfinite_count':s['n_nonfinite'],'all_invalid':s['all_nonfinite'],'constant_after_resampling':s['constant'],'negative_optical_value_count':int((a<0).sum()),'role':'DEVELOPMENT_REUSED' if q.split_group in set(d['y'].split_group) else 'IDENTITY_OPTICS_ONLY_NO_TARGETS'})
        first=d['opt'].sort_values('sample_id').head(4 if ds=='D1' else 1)
        for q in first.itertuples(index=False):
            for ch in (channels if ds=='D2' else ['LP725']):
                a=np.asarray(getattr(q,ch),float); z,_=r.resample_shape(a); label=q.sample_id+' '+ch
                axes[row,0].plot(np.linspace(0,1,len(a)),a,label=label,lw=1); axes[row,1].plot(np.linspace(0,1,256),z,label=label,lw=1)
                examples.extend({'dataset':ds,'sample_id':q.sample_id,'channel':ch,'relative_frame_index':float(t),'shape_value':float(v)} for t,v in zip(np.linspace(0,1,256),z))
        for ax,kind in zip(axes[row],['Observed optical trace','Self-normalized resampling audit']): ax.set_title(ds+' '+kind); ax.set_xlabel('Relative frame index (not seconds)'); ax.set_ylabel('Source intensity (published units)' if ax is axes[row,0] else 'Self min-max shape'); ax.legend(fontsize=6); ax.grid(alpha=.2)
    fig.suptitle('R0 real optical examples: deterministic sample-ID order, no PCE selection\nND is an optical control, not a PL emission spectrum'); fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(r.BASE/'figures'/('R0_real_trajectory_examples.'+ext),dpi=150)
    plt.close(fig)
    details=pd.DataFrame(records); r.csv(details,out/'raw_trace_audit.csv'); r.csv(pd.DataFrame(schemas),out/'input_role_audit.csv'); r.csv(pd.DataFrame(examples),out/'trajectory_examples_resampled.csv')
    summary=details.groupby(['dataset','channel','length','role']).agg(n=('sample_id','size'),nonfinite_values=('nonfinite_count','sum'),all_invalid=('all_invalid','sum'),constant=('constant_after_resampling','sum'),negative_values=('negative_optical_value_count','sum')).reset_index(); r.csv(summary,out/'raw_trace_summary.csv')
    r.dump(out/'D2_raw_F2_reconciliation.json',data['D2']['cache_max_differences'])
    return details

def run(output):
    output=r.safe(output); assert not output.exists(),'Preserve prior attempts'; output.mkdir(parents=True); start=time.perf_counter(); cpu=time.process_time(); c=r.config(); out=output/'results'; out.mkdir(); data=r.load_development()
    optical_audit(data,out); scores=[]; metrics=[]; assignments=[]; checks=[]; cost=[]; pairs=[]; selections=[]; preprocess=[]; reference_copies=[]
    with threadpool_limits(limits=1):
        for ds,d in data.items():
            reference=r.ROOT/'research/pl_public_data_pilot_06_baselines_ablations/results'/('d1_train_oof_predictions.csv' if ds=='D1' else 'd2_dev_oof_predictions.csv')
            refs=['PAIR_ABS','ID_PAIRWISE','EXTRATREES_PL','LGBM_PL','PL_INTENSITY','AUTHOR_KNN_AUX' if ds=='D1' else 'F2_RIDGE_AUX']
            old=pd.read_csv(reference,usecols=r.KEYS+['target','outer_fold']+refs,dtype={'split_group':str}); current=d['y'].merge(d['folds'][['sample_id','outer_fold']],on='sample_id',validate='one_to_one')
            assert set(old.sample_id)==set(current.sample_id) and not old.sample_id.duplicated().any()
            o=old.set_index('sample_id').loc[current.sample_id]; assert np.array_equal(current.outer_fold,o.outer_fold)
            assert np.allclose(current.target,o.target,atol=c['reconciliation']['targets_atol'],rtol=0)
            source_copy=old.drop(columns='target'); r.csv(source_copy,out/(ds+'_P6_reference_predictions_NO_TARGET.csv')); reference_copies.append({'dataset':ds,'path':str(reference),'sha256':r.sha(reference),'role':'DEVELOPMENT_ONLY_SCORING_REFERENCE','excluded_target_column':True})
            for fold in sorted(d['folds'].outer_fold.unique()):
                groups=sorted(d['folds'].loc[d['folds'].outer_fold==fold,'split_group'].unique()); before=time.perf_counter(); sc,model=r.fit_predict_m0(ds,d['features'],d['y'],groups)
                # Match original labeled candidate pool only after target-free inference on every optical row.
                sc=d['y'].loc[d['y'].split_group.isin(groups),r.KEYS].merge(sc,on=r.KEYS,validate='one_to_one'); sc['model_id']='M0_P6_ABS_ERM'; sc['scope']='DEVELOPMENT_REUSED'; sc['outer_fold']=int(fold)
                r.csv(sc,out/f'{ds}_fold{fold}_scores_BEFORE_label_merge.csv')
                maximum=float(abs(sc.score.to_numpy()-old.set_index('sample_id').loc[sc.sample_id,'PAIR_ABS'].to_numpy()).max()); assert maximum<c['reconciliation']['scores_atol']
                for unit,q in sc.groupby('unit_id'):
                    ids=q.sample_id.to_numpy(); k=r.p.top_k(len(q)); a=set(ids[r.p.order(q.score,ids)[:k]]); b=set(ids[r.p.order(old.set_index('sample_id').loc[ids,'PAIR_ABS'],ids)[:k]]); assert a==b
                    selections.extend({'dataset':ds,'unit_id':unit,'sample_id':sid,'selected_top25':sid in a,'P6_selected_top25':sid in b,'outer_fold':int(fold)} for sid in ids)
                checks.append({'dataset':ds,'outer_fold':int(fold),'n_labeled':len(sc),'candidate_ID_set':'EXACT','fold_identity':'EXACT','Top25_ID_sets':'EXACT','max_abs_score_difference':maximum,'status':'PASS'})
                scores.append(sc); metrics.append(r.p.score_groups(d['y'],sc,ds)); cost.append({'dataset':ds,'outer_fold':int(fold),'n_training_groups':len(model.training_groups_),'n_validation_groups':len(groups),'n_features':len(model.pre_.names_),'n_ordered_pairs':model.n_pairs_,'pair_ID_sha256':model.pair_hash_,'wall_seconds':time.perf_counter()-before,'peak_process_MiB':r.p.peak_memory_mib()})
                for role,gg in [('train',model.training_groups_),('validation',groups)]: assignments.extend({'dataset':ds,'outer_fold':int(fold),'role':role,'split_group':v} for v in gg)
                for i,n in enumerate(model.pre_.names_): preprocess.append({'dataset':ds,'outer_fold':int(fold),'feature':n,'train_mean':model.pre_.scaler_.mean_[i],'train_scale':model.pre_.scaler_.scale_[i]})
                audit=model.pair_audit_.copy(); audit['dataset']=ds; audit['outer_fold']=int(fold); pairs.append(audit)
                print(f'{ds} M0 fold={fold}: scores and exact Top25 IDs reconciled',flush=True); del model; gc.collect()
            # Other controls are existing frozen predictions, not newly trained algorithms.
            for mid in refs[1:]:
                sc=old[r.KEYS+['outer_fold',mid]].rename(columns={mid:'score'}).copy(); sc['scope']='DEVELOPMENT_REUSED'; sc['model_id']='P6_REFERENCE_'+mid; scores.append(sc); metrics.append(r.p.score_groups(d['y'],sc,ds))
            r.csv(d['y'],out/(ds+'_evaluation_labels_ONLY.csv')); r.csv(d['folds'],out/(ds+'_canonical_folds.csv')); r.csv(d['history_identity'],out/(ds+'_HISTORY_identity_ONLY.csv'))
    sc=pd.concat(scores,ignore_index=True); mt=pd.concat(metrics,ignore_index=True); units=pd.concat([r.independent(mt[mt.dataset==ds],ds) for ds in data],ignore_index=True); macro=[]; metric_checks=[]
    for (ds,mid),g in units.groupby(['dataset','model_id'],sort=True):
        row={'dataset':ds,'model_id':mid,'scope':'DEVELOPMENT_REUSED','n_independent_groups':len(g),'n_rho_defined':int(g.rho.notna().sum())}
        for m in r.METRICS:
            v=g[m].dropna().to_numpy(); row[m]=v.mean(); rng=np.random.default_rng(c['seed']); b=rng.choice(v,size=(10000,len(v)),replace=True).mean(1); row[m+'_ci95_low'],row[m+'_ci95_high']=np.quantile(b,[.025,.975])
        macro.append(row)
        if mid=='M0_P6_ABS_ERM':
            old=pd.read_csv(r.ROOT/'research/pl_public_data_pilot_06_baselines_ablations/results'/(ds.lower()+'_macro_model_comparison.csv')); ref=old[(old.model_id=='PAIR_ABS')&(old.scope==r.p.config()[ds]['scope_dev'])].iloc[0]
            for m in r.METRICS:
                delta=abs(row[m]-ref[m]); assert delta<c['reconciliation']['metrics_atol']; metric_checks.append({'dataset':ds,'metric':m,'P9_R0_reproduced':row[m],'P6_published':ref[m],'absolute_difference':delta,'status':'PASS'})
    r.csv(sc,out/'oof_predictions.csv'); r.csv(mt,out/'group_metrics.csv'); r.csv(units,out/'independent_group_metrics.csv'); r.csv(pd.DataFrame(macro),out/'dataset_macro_metrics.csv'); r.csv(pd.DataFrame(checks),out/'baseline_reconciliation.csv'); r.csv(pd.DataFrame(metric_checks),out/'baseline_metric_reconciliation.csv'); r.csv(pd.DataFrame(assignments),out/'outer_group_assignments.csv'); r.csv(pd.DataFrame(selections),out/'M0_Top25_selection_identity.csv'); r.csv(pd.concat(pairs,ignore_index=True),out/'M0_training_pair_audit.csv'); r.csv(pd.DataFrame(preprocess),out/'M0_training_scaler_parameters.csv'); r.csv(pd.DataFrame(cost),out/'model_runtime.csv'); r.dump(out/'reference_source_provenance.json',reference_copies)
    tails=[]
    for (ds,mid),g in units.groupby(['dataset','model_id']):
        v=g.top25_gain_pp.to_numpy(); tails.append({'dataset':ds,'model_id':mid,'minimum_gain_pp':v.min(),'minimum_group':g.iloc[int(v.argmin())].split_group,'q05_gain_pp':np.quantile(v,.05),'q10_gain_pp':np.quantile(v,.1),'negative_gain_groups':int((v<0).sum()),'n_groups':len(v),'quantile_interpretation':'DESCRIPTIVE_REUSED_GROUPS'})
    r.csv(pd.DataFrame(tails),out/'failure_tail.csv'); r.csv(mt[(mt.top25_gain_pp<0)|(mt.rho<0)|mt.rho.isna()],out/'negative_and_failure_groups.csv')
    r.dump(out/'NOT_YET_RUN.json',{'M1':'NOT_YET_RUN_R1_NOT_AUTHORIZED','M2':'NOT_YET_RUN_R1_NOT_AUTHORIZED','M3':'NOT_YET_RUN_R1_NOT_AUTHORIZED','historical_evaluation':'NOT_RUN_R0_DEFAULT','paired_deltas_new_vs_M0':'NOT_YET_RUN; no zero placeholders','final_two_complete_replays':'NOT_YET_RUN_R2_NOT_AUTHORIZED'})
    r.dump(output/'run_receipt.json',{'status':'SUCCESS','round':'R0','M0_fits':12,'M1_M2_M3_fits':0,'history_evaluations':0,'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'peak_process_MiB':r.p.peak_memory_mib(),'threads':1,'protocol_sha256':r.sha(r.BASE/'configs/frozen_p9_protocol.json'),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'versions':{n:md.version(n) for n in ['numpy','pandas','scipy','scikit-learn','numba','llvmlite','pytest','matplotlib','tables']}})
    print('R0 development-only M0 reproduction and optical audit SUCCESS; M1-M3 NOT_YET_RUN',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True,type=Path); run(ap.parse_args().output)
