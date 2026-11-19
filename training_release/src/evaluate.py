"""Score trained predictions and compare them to matching archived sample/model/fold IDs."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from release_common import B,fresh_output,load_core,read_json,dump
from dataset import Dataset
def evaluate(args):
    p=load_core();data=Dataset(args.prepared,args.dataset);pred=pd.read_csv(args.predictions,dtype={'split_group':str},float_precision='round_trip');required=set(p.KEYS+['score','model_id','scope','outer_fold'])
    if set(pred.columns)&{'target','pce_mean','PCE_backward','y_true'} or not required<=set(pred.columns):raise ValueError('Supply the target-free prediction table')
    out=fresh_output(args.output);mt=p.score_groups(data.y(),pred,args.dataset);macro,units=p.summarize(mt,args.dataset,read_json(B/'configs/frozen_p6_protocol.json'))
    for name,q in [('selection_unit_metrics',mt),('independent_group_metrics',units),('macro_metrics',macro)]:q.to_csv(out/(name+'.csv'),index=False,float_format='%.17g')
    if args.compare_reference:
        rows=[]
        for (scope,mid,fold),g in pred.groupby(['scope','model_id','outer_fold']):
            history=fold==-1;name=('d1_author_test_predictions.csv' if history else 'd1_train_oof_predictions.csv') if args.dataset=='D1' else ('d2_history_predictions.csv' if history else 'd2_dev_oof_predictions.csv');ref=pd.read_csv(B/'reference_results/P6'/name,float_precision='round_trip').set_index('sample_id')
            if mid not in ref:raise ValueError('No archived model '+mid)
            expected=set(ref.loc[ref.outer_fold==fold].index)
            if set(g.sample_id)!=expected:raise ValueError('Prediction cohort differs from archived model/fold')
            diff=np.abs(g.score.to_numpy()-ref.loc[g.sample_id,mid].to_numpy());rows.append(dict(model_id=mid,scope=scope,outer_fold=int(fold),rows=len(g),max_abs_score_difference=float(diff.max()),within_1e_8=bool(diff.max()<1e-8)))
        pd.DataFrame(rows).to_csv(out/'reference_score_comparison.csv',index=False);dump(out/'reference_comparison_summary.json',{'all_within_1e_8':all(r['within_1e_8'] for r in rows),'fits_replayed':len(rows),'reference_is_used_only_after_prediction':True})
    print('Evaluated frozen prediction scores; zero new fits',out)
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--prepared',type=Path,required=True);a.add_argument('--dataset',choices=['D1','D2'],required=True);a.add_argument('--predictions',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--compare-reference',action='store_true');evaluate(a.parse_args())
