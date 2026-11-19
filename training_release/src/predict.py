"""Predict from a saved frozen-training artifact; no evaluation labels are used."""
import argparse
from pathlib import Path
from release_common import fresh_output,dump
from dataset import Dataset
import portable_models as pm
def run(args):
    import joblib
    pm.dependencies();model=joblib.load(args.model);data=Dataset(args.prepared,model.dataset,load_labels=False);groups=set(args.groups.split(',')) if args.groups else data.role_groups(args.role);out=fresh_output(args.output);pred=model.predict(data,groups);pred.to_csv(out/'scores_NO_TARGET.csv',index=False,float_format='%.17g');dump(out/'prediction_receipt.json',{'model_id':model.model_id,'dataset':model.dataset,'rows':len(pred),'labels_used':False,'groups':sorted(groups)})
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--model',type=Path,required=True);a.add_argument('--prepared',type=Path,required=True);a.add_argument('--groups');a.add_argument('--role',choices=['development','history'],default='development');a.add_argument('--output',type=Path,required=True);run(a.parse_args())
