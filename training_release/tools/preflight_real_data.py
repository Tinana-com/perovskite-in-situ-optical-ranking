"""Audit original training inputs/pair identities without any supervised estimator fit."""
from pathlib import Path
import sys,hashlib,argparse
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from dataset import Dataset
from release_common import load_core,read_json,dump
import numpy as np,pandas as pd
def run(args):
    p=load_core();c=read_json(B/'configs/frozen_p6_protocol.json');original=pd.read_csv(B/'reference_results/P6/training_pair_audit.csv');parameters=pd.read_csv(B/'reference_results/P6/training_preprocessing_parameters.csv');records=[]
    for ds in ['D1','D2']:
        d=Dataset(args.prepared,ds);d.validate_cohort();spec=next(s for s in p.grid() if s['dataset']==ds and s['model_id']=='PAIR_ABS')
        for fold,tr,ev in d.cases():
            y=d.y(tr).sort_values(['split_group','unit_id','sample_id']).reset_index(drop=True);f=d.optical(groups=tr);pre=p.Preprocessor(ds,spec,c).fit(f);x=pd.DataFrame(pre.transform(f),index=f.sample_id).loc[y.sample_id].to_numpy();xx,labels,weights,audit,ids=p.pairs(y,x,ds);digest=hashlib.sha256('\n'.join(ids).encode()).hexdigest();ref=original[(original.dataset==ds)&(original.model_id=='PAIR_ABS')&(original.outer_fold==fold)&(original.scope==c[ds]['scope_dev'])]
            assert len(ref)>0 and ref.pair_ID_sha256.nunique()==1 and digest==ref.pair_ID_sha256.iloc[0] and len(labels)==int(ref.n_ordered_pairs.iloc[0])
            refp=parameters[(parameters.dataset==ds)&(parameters.model_id=='PAIR_ABS')&(parameters.outer_fold==fold)&(parameters.scope==c[ds]['scope_dev'])].dropna(subset=['training_scaler_mean']).set_index('feature').loc[pre.names_]
            means=float(abs(pre.scaler_.mean_-refp.training_scaler_mean.to_numpy()).max());scales=float(abs(pre.scaler_.scale_-refp.training_scaler_scale.to_numpy()).max())
            assert np.allclose(pre.scaler_.mean_,refp.training_scaler_mean,rtol=1e-10,atol=1e-7) and np.allclose(pre.scaler_.scale_,refp.training_scaler_scale,rtol=1e-10,atol=1e-7)
            records.append({'dataset':ds,'outer_fold':int(fold),'training_groups':sorted(tr),'validation_groups':sorted(ev),'group_isolation':not bool(tr&ev),'ordered_pairs':len(labels),'pair_ID_sha256':digest,'pair_ids_equal_archived':True,'max_scaler_mean_abs_difference':means,'max_scaler_scale_abs_difference':scales,'no_supervised_fit':True})
    dump(args.output,{'status':'PASS','folds':records,'all12_outer_fold_training_pairs_match_archive':True,'supervised_model_fits':0,'preprocessing_scalers_only':True});print('PASS: exact pair identities and training-only scalers for all12 outer folds; zero supervised fits',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepared',type=Path,required=True);a.add_argument('--output',type=Path,required=True);run(a.parse_args())
