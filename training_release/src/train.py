"""Train frozen P6 configurations and inherited controls; save target-free OOF scores."""
import argparse,time,traceback,warnings,sys
from pathlib import Path
from release_common import B,fresh_output,dump,read_json,sha,load_core
from dataset import Dataset
import portable_models as pm
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
def save(df,path):df.to_csv(path,index=False,float_format='%.17g',lineterminator='\n')
def build_case(data,train_groups,validation_groups):
    if set(train_groups)&set(validation_groups):raise ValueError('Training and validation groups overlap')
    y=data.y(train_groups);keys=data.labels.loc[data.labels.split_group.isin(validation_groups),load_core().KEYS].copy()
    if set(y.split_group)&set(keys.split_group):raise ValueError('Label group leakage')
    return y,keys
def plan(data,specs,phase):
    cases=[]
    for ph in (['development','history'] if phase=='full' else ['development']):
        for fold,tr,ev in data.cases(ph):
            build_case(data,tr,ev)
            cases.append(dict(dataset=data.name,phase=ph,outer_fold=fold,training_groups=sorted(tr),validation_groups=sorted(ev),models=[s['model_id'] for s in specs],validation_labels_for_fitting=False))
    return cases
def run(args):
    data=Dataset(args.prepared,args.dataset);data.validate_cohort();specs=pm.configuration(args.dataset)
    if args.models!='all':
        requested=args.models.split(',');available={s['model_id'] for s in specs}
        if set(requested)-available:raise ValueError('Unknown frozen model IDs '+str(set(requested)-available))
        specs=[s for s in specs if s['model_id'] in requested]
    out=fresh_output(args.output);cases=plan(data,specs,args.phase);dump(out/'training_plan.json',cases)
    if args.dry_run:dump(out/'run_receipt.json',{'status':'DRY_RUN_NO_MODEL_FITS','model_fits':0,'cases':len(cases)});print('Training plan validated; no fits',flush=True);return
    p=load_core();c=read_json(B/'configs/frozen_p6_protocol.json');start=time.perf_counter();scores=[];metrics=[];statuses=[];lineage=[]
    with threadpool_limits(limits=1):
        for case in cases:
            ph=case['phase'];fold=case['outer_fold'];tr=set(case['training_groups']);ev=set(case['validation_groups']);yt,keys=build_case(data,tr,ev)
            if ph=='history' and not (out/'development_freeze.json').exists():raise ValueError('Development results must freeze before historical scoring')
            for spec in specs:
                mid=spec['model_id'];dest=out/'models'/f'{ph}_fold{fold}'/mid;dest.mkdir(parents=True);begin=time.perf_counter()
                if time.perf_counter()-start>args.max_wall_seconds:
                    statuses.append(dict(phase=ph,outer_fold=fold,model_id=mid,status='NOT_RUN_RESOURCE_LIMIT'));continue
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');model=pm.fit(data,spec,tr);allpred=model.predict(data,ev)
                    pred=keys.merge(allpred,on=p.KEYS,validate='one_to_one');assert len(pred)==len(keys)
                    pred['dataset']=args.dataset;pred['scope']=c[args.dataset]['scope_dev' if ph=='development' else 'scope_test'];pred['phase']=ph;pred['model_id']=mid;pred['outer_fold']=fold
                    # Predictions and selection IDs freeze before evaluation-label projection.
                    save(pred,dest/'scores_BEFORE_label_merge.csv');selected=[]
                    for uid,g in pred.groupby('unit_id',sort=True):selected.extend(g.iloc[p.order(g.score,g.sample_id)[:p.top_k(len(g))]].sample_id.tolist())
                    save(pd.DataFrame({'sample_id':selected}),dest/'Top25_selected_IDs_NO_TARGET.csv');joblib.dump(model,dest/'model.joblib')
                    yy=data.y(ev);mt=p.score_groups(yy,pred,args.dataset);metrics.append(mt);scores.append(pred)
                    audit={'dataset':args.dataset,'phase':ph,'outer_fold':fold,'model_id':mid,'fit_info':model.info,'training_ID_sha256':__import__('hashlib').sha256('\n'.join(sorted(yt.sample_id)).encode()).hexdigest(),'validation_ID_sha256':__import__('hashlib').sha256('\n'.join(sorted(keys.sample_id)).encode()).hexdigest(),'wall_seconds':time.perf_counter()-begin,'warnings':[str(w.message) for w in caught]}
                    # JSON null represents undefined descriptive inner metrics.
                    audit=__import__('json').loads(__import__('json').dumps(audit,default=lambda x:x.item() if hasattr(x,'item') else str(x)).replace('NaN','null'));dump(dest/'fit_audit.json',audit);lineage.append(audit)
                    statuses.append(dict(phase=ph,outer_fold=fold,model_id=mid,status='SUCCESS',wall_seconds=time.perf_counter()-begin));print(args.dataset,ph,fold,mid,'SUCCESS',flush=True)
                except Exception as exc:
                    (dest/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');statuses.append(dict(phase=ph,outer_fold=fold,model_id=mid,status='FAILED_PRESERVED',error=repr(exc)));print(args.dataset,ph,fold,mid,'FAILED_PRESERVED',repr(exc),flush=True)
                save(pd.DataFrame(statuses),out/'fit_status.csv')
            if ph=='development' and fold==max(q['outer_fold'] for q in cases if q['phase']=='development'):
                dump(out/'development_freeze.json',{'protocol_sha256':sha(B/'configs/frozen_p6_protocol.json'),'model_grid_sha256':sha(B/'configs/model_grid.csv'),'requested_model_ids':[s['model_id'] for s in specs],'history_labels_used_for_selection':False,'parameters_changed':False,'completed_fit_status_sha256':sha(out/'fit_status.csv')})
    if scores:
        sc=pd.concat(scores,ignore_index=True);mt=pd.concat(metrics,ignore_index=True)
        for ph in sorted({case['phase'] for case in cases}):
            groups=data.role_groups(ph);yy=data.y(groups);fm={str(g):int(f) for g,f in data.labels.loc[data.labels.split_group.isin(groups),['split_group','outer_fold']].drop_duplicates().to_numpy()};mt=pd.concat([mt,p.random_groups(yy,c[args.dataset]['scope_dev' if ph=='development' else 'scope_test'],args.dataset,fm)],ignore_index=True)
        save(sc,out/'predictions_BEFORE_label_merge.csv');save(mt,out/'selection_unit_metrics.csv');macro,units=p.summarize(mt,args.dataset,c);save(macro,out/'macro_metrics.csv');save(units,out/'independent_group_metrics.csv');scored=sc.merge(data.y(),on=p.KEYS,validate='many_to_one');save(scored,out/'predictions_WITH_scoring_targets.csv');save(mt[(mt.top25_gain_pp<0)|(mt.rho<0)|mt.rho.isna()],out/'negative_and_failure_groups.csv')
    status='COMPLETE' if all(x['status']=='SUCCESS' for x in statuses) else 'COMPLETE_WITH_FAILURES';dump(out/'run_receipt.json',{'status':status,'supervised_fits_attempted':sum(x['status']!='NOT_RUN_RESOURCE_LIMIT' for x in statuses),'successful':sum(x['status']=='SUCCESS' for x in statuses),'wall_seconds':time.perf_counter()-start,'threads':1,'seed':20261019,'source_protocol_sha256':sha(B/'configs/frozen_p6_protocol.json'),'phase':args.phase,'history_status':'PREVIOUSLY_INSPECTED','validation_targets_used_for_fit':False});print(status,out,flush=True)
    if status!='COMPLETE':raise SystemExit(1)
def parser():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--prepared',type=Path,required=True);a.add_argument('--dataset',choices=['D1','D2'],required=True);a.add_argument('--models',default='all',help='all, or comma-separated predeclared model IDs; no new parameter search');a.add_argument('--phase',choices=['development','full'],default='development');a.add_argument('--output',type=Path,required=True);a.add_argument('--max-wall-seconds',type=int,default=7200);a.add_argument('--dry-run',action='store_true');return a
if __name__=='__main__':run(parser().parse_args())
