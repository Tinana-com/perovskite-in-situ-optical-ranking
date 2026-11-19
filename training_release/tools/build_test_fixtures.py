"""Small excerpts of actual public data for unit tests; never a research cohort."""
from pathlib import Path
import sys
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from dataset import Dataset
from release_common import dump,sha
import numpy as np
source=Path(sys.argv[1]);out=B/'tests/fixtures';out.mkdir(exist_ok=True);manifest={'test_fixture_only':True,'datasets':{},'raw_inputs':{},'files':{}}
for ds in ['D1','D2']:
    data=Dataset(source,ds);dev=data.meta[data.meta.role=='development'];groups=[]
    for fold in sorted(dev.outer_fold.unique())[:4 if ds=='D1' else 3]:
        groups.append(sorted(dev.loc[dev.outer_fold==fold,'split_group'].unique())[0])
    groups.append(sorted(data.role_groups('history'))[0]);sel=data.meta.split_group.isin(groups)
    if ds=='D2':
        units=[]
        for group in groups:units+=sorted(data.labels.loc[data.labels.split_group==group,'unit_id'].unique())[:2]
        sel&=data.meta.unit_id.isin(units)
    ids=set(data.meta.loc[sel,'sample_id']);d=out/ds;d.mkdir(exist_ok=True)
    data.meta[sel].to_csv(d/'optical_identity.csv',index=False);data.labels[data.labels.sample_id.isin(ids)].to_csv(d/'labels.csv',index=False,float_format='%.17g');np.savez_compressed(d/'features.npz',**{k:v[sel] for k,v in data.features.items()})
    if ds=='D1':np.savez_compressed(d/'raw_LP725.npz',LP725=data.traces[sel])
    manifest['datasets'][ds]=data.spec|{'test_selection':'first substrate/date of first author/date folds; first2 eligible experiment IDs per date; optical rows preserved; no outcome-magnitude selection'}
manifest['files']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='data_manifest.json'};dump(out/'data_manifest.json',manifest)
print('Saved small real-data test excerpts; training CLI rejects them as incomplete research cohorts')
