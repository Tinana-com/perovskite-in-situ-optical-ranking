"""Recompute the taskbook's post-hoc P3 claim before starting P4 fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ranking_metrics import group_metrics
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parents[1]

def audit(out):
    out=Path(out).resolve()
    if not out.is_relative_to(BASE): raise ValueError('Audit writes restricted to P4')
    out.mkdir(parents=True,exist_ok=True)
    p3=ROOT/'research/pl_public_data_pilot_03'
    if hashlib.sha256((p3/'PL_P3_RESULTS_FOR_CHATGPT.zip').read_bytes()).hexdigest()!='b6671665a554259202ac6aa4138d0709d53f537bcd7752d2f3cb710d45f94eb5': raise ValueError('P3 archive version differs from taskbook')
    rows=[]; groups=[]
    for scope,file in [('EXPLORATORY_REUSED_DEV_DATES','outer_oof_predictions.csv'),('PREVIOUSLY_INSPECTED_HOLDOUT','heldout_predictions.csv')]:
        frame=pd.read_csv(p3/'results'/file)
        for mid in ['B_relative_extra_trees','C_two_level']:
            for rounding in ['floor','ceil']:
                parts=[]
                for (date,exp),q in frame[frame.model_id==mid].groupby(['date','expID']):
                    if len(q)<4: continue
                    result=group_metrics(q.y_true,q.y_pred,q.sample_id,rounding)
                    parts.append(result|{'date':int(date),'expID':exp})
                    groups.append(result|{'scope':scope,'model_id':mid,'rounding':rounding,'date':int(date),'expID':exp})
                d=pd.DataFrame(parts)
                rows.append({'scope':scope,'model_id':mid,'rounding':rounding,'n_experiments':len(d),'n_patches':int(d.n.sum()),
                    'experiment_equal_rho':float(d.rho.mean()),'positive_rho_fraction':float((d.rho>0).mean()),
                    'experiment_equal_top25_gain_pp':float(d.top25_gain_pp.mean()),'date_equal_rho':float(d.groupby('date').rho.mean().mean()),
                    'date_equal_top25_gain_pp':float(d.groupby('date').top25_gain_pp.mean().mean()),'undefined_rho':int(d.rho.isna().sum())})
    table=pd.DataFrame(rows); table.to_csv(out/'P3_RECONCILIATION.csv',index=False,float_format='%.12g')
    pd.DataFrame(groups).to_csv(out/'P3_RECONCILIATION_BY_EXPERIMENT.csv',index=False,float_format='%.12g')
    target={('EXPLORATORY_REUSED_DEV_DATES','B_relative_extra_trees'):(.236,.554),('PREVIOUSLY_INSPECTED_HOLDOUT','B_relative_extra_trees'):(.409,1.100),
        ('EXPLORATORY_REUSED_DEV_DATES','C_two_level'):(.265,.636),('PREVIOUSLY_INSPECTED_HOLDOUT','C_two_level'):(.438,1.075)}
    matches=[]
    for row in table.itertuples():
        rr,gg=target[(row.scope,row.model_id)]
        matches.append({'scope':row.scope,'model_id':row.model_id,'rounding':row.rounding,'rho_delta':row.experiment_equal_rho-rr,
            'gain_delta':row.experiment_equal_top25_gain_pp-gg,'matches_approx_claim':abs(row.experiment_equal_rho-rr)<.002 and abs(row.experiment_equal_top25_gain_pp-gg)<.01})
    status={'P3_zip_matches_taskbook':True,'comparisons':matches,'status':'RECONCILED' if all(any(x['matches_approx_claim'] for x in matches if (x['scope'],x['model_id'])==key) for key in target) else 'STOP_EXPLAIN_METRIC_MISMATCH'}
    (out/'P3_RECONCILIATION.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(table.to_string(index=False)); print(status['status'],flush=True)
    if status['status']!='RECONCILED': raise ValueError('Explain P3 metric discrepancy before proceeding')
    return table

if __name__=='__main__': audit(BASE/'results')
