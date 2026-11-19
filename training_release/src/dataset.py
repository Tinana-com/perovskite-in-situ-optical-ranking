"""Prepared optical inputs, labels and frozen group folds kept as separate objects."""
from pathlib import Path
import numpy as np,pandas as pd
from release_common import read_json,sha,load_core
KEYS=['sample_id','unit_id','split_group','patchID']
class Dataset:
    def __init__(self,prepared,name,verify=True,load_labels=True):
        self.root=Path(prepared).resolve();self.name=name;self.manifest=read_json(self.root/'data_manifest.json');self.spec=self.manifest['datasets'][name];d=self.root/name
        if verify:
            for rel,h in self.manifest['files'].items():
                if rel.startswith(name+'/') and (load_labels or not rel.endswith('/labels.csv')) and sha(self.root/rel)!=h:raise ValueError('Prepared input hash changed '+rel)
        self.meta=pd.read_csv(d/'optical_identity.csv',dtype={'split_group':str},float_precision='round_trip');self.labels=pd.read_csv(d/'labels.csv',dtype={'split_group':str},float_precision='round_trip') if load_labels else pd.DataFrame(columns=KEYS+['target']);self.features=dict(np.load(d/'features.npz',allow_pickle=False));self.traces=None
        if name=='D1':self.traces=np.load(d/'raw_LP725.npz',allow_pickle=False)['LP725']
        assert self.meta.sample_id.is_unique and self.labels.sample_id.is_unique
        assert set(self.labels.sample_id)<=set(self.meta.sample_id)
    def optical(self,window='FULL',groups=None):
        f=self.meta[KEYS].copy();cols=self.spec['feature_names'];a=self.features[window]
        if len(f)!=len(a):raise ValueError('Optical identity misalignment')
        for i,col in enumerate(cols):f[col]=a[:,i]
        return f if groups is None else f[f.split_group.isin(set(groups))].reset_index(drop=True)
    def raw(self,groups=None):
        if self.name!='D1':raise ValueError('Only D1 full traces in the prepared archive; D2 raw H5 is read separately for P9')
        f=self.meta[KEYS].copy();f['LP725']=list(self.traces)
        return f if groups is None else f[f.split_group.isin(set(groups))].reset_index(drop=True)
    def rules(self,groups=None):
        if self.name!='D2':raise ValueError('D2 rule matrix only')
        f=self.meta[KEYS].copy()
        for i,col in enumerate(self.spec['rule_names']):f[col]=self.features['RULES'][:,i]
        return f if groups is None else f[f.split_group.isin(set(groups))].reset_index(drop=True)
    def y(self,groups=None):
        f=self.labels[KEYS+['target']].copy();return f if groups is None else f[f.split_group.isin(set(groups))].reset_index(drop=True)
    def role_groups(self,role):return set(self.meta.loc[self.meta.role==role,'split_group'])
    def validate_cohort(self):
        expected=(780,349,26,12,1,0) if self.name=='D1' else (4707,1150,14,3,33,2)
        a=self.labels[self.labels.role=='development'];b=self.labels[self.labels.role=='history']
        actual=(len(a),len(b),a.split_group.nunique(),b.split_group.nunique(),int((a.target==0).sum()),int((b.target==0).sum()))
        if actual!=expected:raise ValueError('Prepared cohort differs from the frozen research cohort')
        if set(a.split_group)&set(b.split_group) or self.labels.groupby('split_group').outer_fold.nunique().max()!=1:raise ValueError('Prepared group/fold isolation failed')
        if set(a.outer_fold)!=set(range(5 if self.name=='D1' else 7)):raise ValueError('Outer folds changed')
        if self.name=='D2' and (a.unit_id.nunique(),b.unit_id.nunique())!=(158,36):raise ValueError('D2 experiment count changed')
        return True
    def cases(self,phase='development'):
        dev=self.role_groups('development')
        if phase=='history':yield -1,dev,self.role_groups('history');return
        folds=self.labels[self.labels.role=='development'][['split_group','outer_fold']].drop_duplicates()
        for fold in sorted(folds.outer_fold.unique()):
            ev=set(folds.loc[folds.outer_fold==fold,'split_group']);yield int(fold),dev-ev,ev
    def legacy(self):
        dev=self.role_groups('development');hist=self.role_groups('history');y=self.y(dev);p=load_core()
        folds=y[KEYS].merge(self.labels[['sample_id','outer_fold']],on='sample_id',validate='one_to_one')
        return dict(opt=self.raw(dev) if self.name=='D1' else self.optical(groups=dev),y=y,testopt=self.raw(hist) if self.name=='D1' else self.optical(groups=hist),testy_SCORING_ONLY=self.y(hist),folds=folds,source_rows=1129 if self.name=='D1' else 11136,optical_rows=len(self.meta),train_groups=len(dev),test_groups=len(hist),schema_columns=self.spec['schema_columns'])
