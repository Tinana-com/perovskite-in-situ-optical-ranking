"""One R1 execution of four frozen models; no history evaluation or selection."""
import sys,time,json,argparse,gc,threading,traceback,os
from pathlib import Path
import r1_core as c
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

class MemoryWatch:
    """Sample process RSS, not an invented exact per-model allocation peak."""
    def __enter__(self):
        self.stop=threading.Event(); self.peak=0; self.process=psutil.Process()
        def watch():
            while not self.stop.is_set(): self.peak=max(self.peak,self.process.memory_info().rss); self.stop.wait(.1)
        self.thread=threading.Thread(target=watch,daemon=True); self.thread.start(); return self
    def __exit__(self,*args): self.peak=max(self.peak,self.process.memory_info().rss); self.stop.set(); self.thread.join()

def run(output):
    output=c.r.safe(output); assert not output.exists(),'Preserve previous attempt; use a new path'; output.mkdir(parents=True); out=output/'results'; out.mkdir(); parameters=output/'model_parameters'; parameters.mkdir()
    start=time.perf_counter(); cpu=time.process_time(); protocol=c.r.config(); execution=json.loads((c.R1/'configs/actual_execution_config.json').read_text()); probe=json.loads((c.R1/'RUN_RECEIPTS/technical_probe_receipt.json').read_text()); assert probe['status']=='PASS'
    assert c.r.sha(c.BASE/'configs/frozen_p9_protocol.json')==execution['frozen_protocol_sha256']
    files={f.relative_to(c.BASE).as_posix():c.r.sha(f) for f in [c.BASE/'src'/n for n in ['r1_core.py','run_r1.py','smoke_r1.py','r0_core.py','p6_core.py','p4_reference.py','p5_reference.py','ranking_metrics.py']]+[c.BASE/'configs/frozen_p9_protocol.json',c.BASE/'configs/model_matrix.csv',c.R1/'configs/actual_execution_config.json']}
    c.r.dump(output/'source_config_freeze.json',files)
    data=c.r.load_development(); all_scores=[]; all_metrics=[]; costs=[]; lineage=[]; preprocessing=[]; selection_rows=[]; status_rows=[]; pairaudits=[]; reconciliation=[]; label_audits=[]
    with threadpool_limits(limits=1):
        for ds,d in data.items():
            print(f'R1 {ds}: prepare actual optical inputs',flush=True); arrays,counters=c.prepare_optical(ds,d['opt']); channels=protocol['M2']['channels'][ds]; shapes,shape_names=c.shape_features(arrays,channels)
            counters['dataset']=ds; c.r.csv(counters,out/(ds+'_trace_preprocessing_audit.csv')); c.r.csv(d['folds'],out/(ds+'_canonical_folds.csv')); c.r.csv(d['y'],out/(ds+'_evaluation_labels_ONLY.csv'))
            featuremeta=d['opt'][c.KEYS].copy()
            for ch in channels: featuremeta[ch+'_normalized_trace_SHA256']=[c.array_hash(a) for a in arrays[ch]]
            c.r.csv(featuremeta,out/(ds+'_optical_identity_and_channel_hashes.csv')); c.r.csv(pd.DataFrame(shapes,columns=shape_names).assign(sample_id=d['opt'].sample_id.to_numpy()),out/(ds+'_frozen_M1_shape_features.csv'))
            for fold in sorted(d['folds'].outer_fold.unique()):
                val=sorted(d['folds'].loc[d['folds'].outer_fold==fold,'split_group'].unique()); tr,ev,yy=c.split_inputs(ds,d['features'],d['y'],val); rocket=None; reference_pair=None
                for mid in ['M0','M1','M2','M3']:
                    prior=execution['prior_R0_main_wall_seconds']+probe['wall_seconds']; elapsed=time.perf_counter()-start
                    if prior+elapsed>=execution['budget_total_seconds']:
                        status_rows.append({'dataset':ds,'outer_fold':int(fold),'model_id':mid,'status':'NOT_RUN_RESOURCE','reason':'Cumulative frozen CPU wall budget exhausted; no parameter change'}); continue
                    begin=time.perf_counter(); fitcpu=time.process_time(); rocket_wall=0.; print(f'R1 {ds} fold={fold} {mid} START',flush=True)
                    try:
                        with MemoryWatch() as memory:
                            if mid=='M2':
                                before=time.perf_counter(); rocket=c.Rocket(ds).fit(arrays,tr,d['features'].sample_id.to_numpy(),d['features'].split_group.to_numpy()); rocket_wall=time.perf_counter()-before; rocket.save(parameters/f'{ds}_fold{fold}_MiniRocket_train_ONLY.npz')
                                lineage.append({'dataset':ds,'outer_fold':int(fold),'used_by':['M2','M3'],'training_groups':rocket.training_groups_,'validation_groups':val,'training_ID_hash':c.id_hash(rocket.training_ids_),'training_optical_rows':len(tr),'training_input_hashes':rocket.training_input_hashes_,'channel_order':rocket.channels,'parameter_hash':rocket.parameter_hash_,'fit_wall_seconds':rocket_wall,'fit_on_validation':False,'Numba_seed':20261019})
                            scored,fitted=c.fit_case(ds,mid,d['features'],d['y'],val,arrays,shapes,rocket)
                        info=fitted['info']; assert info['n_features']==execution['new_feature_dimensions'][ds][mid]
                        if reference_pair is None: reference_pair=info['pair_ID_sha256']
                        assert info['pair_ID_sha256']==reference_pair,'Feature variants must have exact original pair IDs'
                        labeled=d['y'].loc[d['y'].split_group.isin(val),c.KEYS]; scored=labeled.merge(scored,on=c.KEYS,validate='one_to_one'); scored['dataset']=ds; scored['model_id']=mid; scored['scope']='DEVELOPMENT_REUSED'; scored['outer_fold']=int(fold); scored['fit_status']=info['fit_status']
                        # Save every optical prediction and selected region BEFORE target join.
                        selected=[]
                        for unit,q in scored.groupby('unit_id',sort=True):
                            ids=q.sample_id.to_numpy(); wanted=set(ids[c.r.p.order(q.score,ids)[:c.r.p.top_k(len(q))]])
                            selected.extend({'dataset':ds,'model_id':mid,'outer_fold':int(fold),'unit_id':unit,'sample_id':sid,'selected_top25':sid in wanted} for sid in ids)
                        c.r.csv(scored,out/f'{ds}_fold{fold}_{mid}_scores_BEFORE_label_merge.csv'); c.r.csv(pd.DataFrame(selected),out/f'{ds}_fold{fold}_{mid}_selected_BEFORE_label_merge.csv'); selection_rows.extend(selected)
                        c.save_fitted(parameters/f'{ds}_fold{fold}_{mid}_ranker.npz',fitted)
                        audit=info|{'outer_fold':int(fold),'model_total_wall_seconds':time.perf_counter()-begin,'model_CPU_seconds':time.process_time()-fitcpu,'rss_max_sampled_MiB':memory.peak/1024**2,'RSS_sampling_period_seconds':.1,'process_peak_MiB_to_date':c.r.p.peak_memory_mib(),'shared_Rocket_fit_wall_seconds':rocket_wall,'Rocket_fit_reuse':'M3 reuses M2 same-fold training-only transform; no cross-fold reuse' if mid=='M3' else 'NOT_APPLICABLE'}
                        c.r.dump(parameters/f'{ds}_fold{fold}_{mid}_audit.json',audit); costs.append({k:v for k,v in audit.items() if not isinstance(v,(list,dict))})
                        for i,n in enumerate(info['feature_names']): preprocessing.append({'dataset':ds,'outer_fold':int(fold),'model_id':mid,'feature':n,'train_mean':fitted['scaler'].mean_[i],'train_scale':fitted['scaler'].scale_[i]})
                        pa=fitted['pair_audit'].copy(); pa['dataset']=ds; pa['outer_fold']=int(fold); pa['model_id']=mid; pa['pair_ID_sha256']=reference_pair; pairaudits.append(pa)
                        # Audit label interventions for each full outer fold at the only supervised boundary.
                        for action in ['zero','shuffle','mask']:
                            altered=d['y'].copy(); vi=altered.split_group.isin(val); altered.loc[vi,'target']=0. if action=='zero' else np.nan if action=='mask' else np.random.default_rng(41).permutation(altered.loc[vi,'target'].to_numpy()); tt,ee,newy=c.split_inputs(ds,d['features'],altered,val)
                            assert np.array_equal(tr,tt) and np.array_equal(ev,ee) and yy.equals(newy)
                            label_audits.append({'dataset':ds,'outer_fold':int(fold),'model_id':mid,'intervention':action,'training_labels_exactly_unchanged':True,'training_optical_ID_hash':info['training_sample_ID_sha256'],'validation_optical_ID_hash':info['validation_sample_ID_sha256'],'test_type':'ALL_OUTER_FOLDS_SUPERVISED_BOUNDARY; full pipeline refits additionally in technical_probe','current_validation_target_used':False})
                        if mid=='M0':
                            old=pd.read_csv(c.BASE/'results/oof_predictions.csv',dtype={'split_group':str}); old=old[(old.model_id=='M0_P6_ABS_ERM')&old.sample_id.isin(scored.sample_id)].set_index('sample_id'); diff=float(abs(scored.score.to_numpy()-old.loc[scored.sample_id,'score'].to_numpy()).max()); assert diff<protocol['reconciliation']['scores_atol']
                            for unit,q in scored.groupby('unit_id'):
                                ids=q.sample_id.to_numpy(); k=c.r.p.top_k(len(q)); assert set(ids[c.r.p.order(q.score,ids)[:k]])==set(ids[c.r.p.order(old.loc[ids,'score'],ids)[:k]])
                            reconciliation.append({'dataset':ds,'outer_fold':int(fold),'score_max_abs_diff_R0':diff,'candidate_and_fold_IDs':'EXACT','Top25_ID_sets':'EXACT','status':'PASS'})
                        all_scores.append(scored); all_metrics.append(c.r.p.score_groups(d['y'],scored,ds)); status_rows.append({'dataset':ds,'outer_fold':int(fold),'model_id':mid,'status':info['fit_status'],'reason':' | '.join(info['warnings']) or 'Frozen fit converged'})
                        print(f'R1 {ds} fold={fold} {mid} {info["fit_status"]} iterations={info["solver_iterations"]} seconds={time.perf_counter()-begin:.2f} RSSsample={memory.peak/1024**2:.1f}MiB',flush=True)
                        del fitted,scored; gc.collect()
                    except Exception as error:
                        status_rows.append({'dataset':ds,'outer_fold':int(fold),'model_id':mid,'status':'FAILED_PROGRAM','reason':repr(error)}); (output/f'{ds}_fold{fold}_{mid}_failure.txt').write_text(traceback.format_exc(),encoding='utf-8'); print(traceback.format_exc(),flush=True)
                        # Preserve attempt and stop actual program bugs, never select a new protocol.
                        c.r.csv(pd.DataFrame(status_rows),out/'model_status.csv'); raise
                c.r.csv(pd.DataFrame(status_rows),out/'model_status.csv'); c.r.csv(pd.DataFrame(costs),out/'runtime_and_memory.csv'); c.r.dump(out/'MiniRocket_training_lineage.json',lineage)
                del rocket; gc.collect()
            del arrays,shapes; gc.collect()
    for ds,d in data.items():
        old=pd.read_csv(c.BASE/'results/oof_predictions.csv',dtype={'split_group':str}); refs=old[old.sample_id.isin(d['y'].sample_id)&old.model_id.str.startswith('P6_REFERENCE_')].copy(); refs['dataset']=ds; refs['fit_status']='FROZEN_HISTORICAL_OOF_REFERENCE'; assert all(len(g)==len(d['y']) and set(g.sample_id)==set(d['y'].sample_id) for _,g in refs.groupby('model_id')); all_scores.append(refs); all_metrics.append(c.r.p.score_groups(d['y'],refs,ds))
    scores=pd.concat(all_scores,ignore_index=True); metrics=pd.concat(all_metrics,ignore_index=True); c.r.csv(scores,out/'oof_predictions_BEFORE_label_merge.csv'); c.r.csv(metrics,out/'group_metrics.csv'); c.r.csv(pd.DataFrame(selection_rows),out/'Top25_selected_IDs_NO_TARGET.csv'); c.r.csv(pd.DataFrame(reconciliation),out/'M0_reconciliation_with_R0.csv'); c.r.csv(pd.DataFrame(label_audits),out/'all_outer_label_interventions.csv'); c.r.csv(pd.DataFrame(preprocessing),out/'train_ONLY_scaler_parameters.csv'); c.r.csv(pd.concat(pairaudits,ignore_index=True),out/'training_pair_audit.csv')
    assert all(c.r.sha(c.BASE/k)==v for k,v in files.items()),'Executable source changed during formal run'
    rec={'status':'SUCCESS' if all(x['status']=='SUCCESS' for x in status_rows) else 'COMPLETE_WITH_FLAGGED_BRANCHES','round':'R1','wall_seconds':time.perf_counter()-start,'CPU_seconds':time.process_time()-cpu,'process_peak_MiB':c.r.p.peak_memory_mib(),'fit_count_main':len(costs),'MiniRocket_train_group_fits':len(lineage),'MiniRocket_channel_fits':sum(len(q['channel_order']) for q in lineage),'threads':1,'historical_evaluations':0,'new_search':False,'R2_full_replays':'NOT_AUTHORIZED','protocol_sha256':execution['frozen_protocol_sha256'],'main_source_config_unchanged':True,'prior_R0_wall':execution['prior_R0_main_wall_seconds'],'technical_probe_wall':probe['wall_seconds'],'total_accounted_wall':execution['prior_R0_main_wall_seconds']+probe['wall_seconds']+time.perf_counter()-start}
    c.r.dump(output/'run_receipt.json',rec); print(json.dumps(rec),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True,type=Path); run(ap.parse_args().output)
