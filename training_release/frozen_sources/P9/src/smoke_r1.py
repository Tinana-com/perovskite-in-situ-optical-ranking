"""One bounded technical probe on training subsets, not performance selection."""
import time,json,sys
import r1_core as c
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
def probe():
    start=time.perf_counter(); cpu=time.process_time(); data=c.r.load_development(); rows=[]; lineage=[]
    with threadpool_limits(limits=1):
        for ds,d in data.items():
            fold=d['folds'].outer_fold.min(); val=set(d['folds'].loc[d['folds'].outer_fold==fold,'split_group'])
            # deterministic IDs, no target values used to pick optical examples
            raw=d['opt']; train=raw[~raw.split_group.isin(val)].sort_values('sample_id').head(96); ev=raw[raw.split_group.isin(val)].sort_values('sample_id').head(32); opt=pd.concat([train,ev],ignore_index=True)
            ids=opt.sample_id; f=d['features'].set_index('sample_id').loc[ids].reset_index(); f=f[c.KEYS+(c.r.p.p5.STATS if ds=='D1' else c.r.p.p4.F2)]; y=d['y'][d['y'].sample_id.isin(ids)].copy()
            arrays,_=c.prepare_optical(ds,opt); shapes,_=c.shape_features(arrays,c.r.config()['M2']['channels'][ds]); tr=np.flatnonzero(~f.split_group.isin(val)); ve=np.flatnonzero(f.split_group.isin(val))
            a=c.Rocket(ds).fit(arrays,tr,ids.to_numpy(),f.split_group.to_numpy()); b=c.Rocket(ds).fit(arrays,tr,ids.to_numpy(),f.split_group.to_numpy())
            assert a.parameter_hash_==b.parameter_hash_ and np.array_equal(a.transform(arrays,ve),b.transform(arrays,ve))
            lineage.append({'dataset':ds,'train_rows':len(tr),'validation_rows':len(ve),'channels':a.channels,'training_groups':a.training_groups_,'validation_groups':sorted(val),'parameter_hash_first':a.parameter_hash_,'parameter_hash_second':b.parameter_hash_,'Numba_compiled_seed_repeat_byte_identical':True,'validation_fit':False})
            for mid in ['M0','M1','M2','M3']:
                baseline,fit=c.fit_case(ds,mid,f,y,val,arrays,shapes); assert np.isfinite(baseline.score).all()
                for action in ['zero','shuffle','mask']:
                    altered=y.copy(); isval=altered.split_group.isin(val)
                    altered.loc[isval,'target']=0. if action=='zero' else np.nan if action=='mask' else np.random.default_rng(37).permutation(altered.loc[isval,'target'].to_numpy())
                    scored,again=c.fit_case(ds,mid,f,altered,val,arrays,shapes)
                    assert np.array_equal(baseline.score.to_numpy(),scored.score.to_numpy())
                    assert fit['info']['coefficient_sha256']==again['info']['coefficient_sha256']
                    rows.append({'dataset':ds,'model_id':mid,'intervention':action,'n_train_optical':len(tr),'n_validation_optical':len(ve),'n_features':fit['info']['n_features'],'score_byte_identical':True,'coeff_byte_identical':True,'fit_status':fit['info']['fit_status'],'altered_targets_used_for_fit':False})
                print(f'PROBE {ds} {mid}: dimensions/finiteness and zero/shuffle/mask refits PASS',flush=True)
            # Constant and invalid traces are separate explicitly synthetic fixtures, not main data.
            z,status=c.r.resample_shape([np.nan,0.,np.inf,0.]); assert np.isfinite(z).all() and status['constant']
            const={ch:np.zeros((5,256),np.float32) for ch in a.channels}; cp=c.Rocket(ds).fit(const,np.arange(4),np.array(list('abcde')),np.array(['fixture_train']*4+['fixture_val'])); out=cp.transform(const,np.array([4])); assert out.shape==(1,len(a.channels)*1008) and np.isfinite(out).all()
    c.r.csv(pd.DataFrame(rows),c.R1/'RUN_RECEIPTS/technical_probe_label_interventions.csv'); c.r.dump(c.R1/'RUN_RECEIPTS/technical_probe_lineage.json',lineage)
    rec={'status':'PASS','purpose':'Technical execution and leakage only, not model selection','real_data_source':'original outer fold0 training optical subset96 and validation32 per dataset; ID-order selection only','full_pipeline_intervention_tests':len(rows),'compiled_RNG_determinism':'parameter and transform byte equality PASS','constant_and_NaN_fixtures':'PASS; separate from real results','models_or_protocol_selected_by_probe':False,'parameters_changed':False,'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'process_peak_MiB':c.r.p.peak_memory_mib(),'threads':1,'largest_probe_features':3036}
    c.r.dump(c.R1/'RUN_RECEIPTS/technical_probe_receipt.json',rec); print(json.dumps(rec),flush=True)
if __name__=='__main__': probe()
