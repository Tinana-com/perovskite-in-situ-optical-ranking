"""Verify real public H5 inputs and prepare exact frozen optical model inputs."""
from pathlib import Path
import argparse,time
from release_common import B,load_core,sha,dump,read_json,fresh_output
import numpy as np,pandas as pd
KEYS=['sample_id','unit_id','split_group','patchID']
def verify_file(path,spec):
    path=Path(path).resolve()
    if not path.is_file():raise FileNotFoundError(path)
    if path.stat().st_size!=spec['bytes'] or sha(path)!=spec['sha256']:raise ValueError('Frozen source size/SHA256 mismatch: '+str(path))
    return path
def save_frame(f,p):f.to_csv(p,index=False,float_format='%.17g',lineterminator='\n')
def prepare(args):
    out=fresh_output(args.output);p=load_core();c=read_json(B/'configs/frozen_p6_protocol.json');receipt={'model_fits':0,'datasets':{},'raw_inputs':{},'seed':c['seed']};start=time.perf_counter()
    if args.dataset in ['D1','both']:
        if not args.d1_train or not args.d1_test:raise ValueError('D1 requires --d1-train and --d1-test')
        pathtr=verify_file(args.d1_train,c['D1']['train_file']);pathte=verify_file(args.d1_test,c['D1']['test_file']);rawtr=pd.read_hdf(pathtr,'df');rawte=pd.read_hdf(pathte,'df')
        if set(rawtr.substrateID)&set(rawte.substrateID):raise ValueError('Substrate overlap')
        if (len(rawtr),len(rawte),rawtr.substrateID.nunique(),rawte.substrateID.nunique())!=(780,349,26,12) or rawtr.groupby('substrateID').foldNo.nunique().max()!=1 or set(rawtr.foldNo)!=set(range(5)):raise ValueError('D1 frozen cohorts/folds differ')
        meta=[];traces=[];labels=[]
        for role,raw,path in [('development',rawtr,pathtr),('history',rawte,pathte)]:
            ids=p.d1_identity(raw);m=ids.copy();m['role']=role;m['outer_fold']=raw.foldNo.to_numpy(int) if role=='development' else -1;meta.append(m)
            y=ids.copy();y['target']=raw.pce_mean.to_numpy(float);y['role']=role;y['outer_fold']=m.outer_fold;labels.append(y);traces.extend([np.asarray(a,float) for a in raw.LP725]);receipt['raw_inputs']['D1_'+role]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
        meta=pd.concat(meta,ignore_index=True);y=pd.concat(labels,ignore_index=True);a=np.vstack(traces);assert a.shape==(1129,719) and np.isfinite(y.target).all()
        d=out/'D1';d.mkdir();save_frame(meta,d/'optical_identity.csv');save_frame(y,d/'labels.csv');np.savez_compressed(d/'raw_LP725.npz',LP725=a)
        opt=meta[KEYS].copy();opt['LP725']=list(a)
        matrices={window:p.d1_features(opt,window)[p.p5.STATS].to_numpy() for window in ['FULL','EARLY25','LATE25']};np.savez_compressed(d/'features.npz',**matrices)
        receipt['datasets']['D1']={'development_rows':780,'history_rows':349,'development_groups':26,'history_groups':12,'feature_names':p.p5.STATS,'schema_columns':list(rawtr.columns),'label_zeros':int((y.target==0).sum())}
    if args.dataset in ['D2','both']:
        if not args.d2_h5:raise ValueError('D2 requires --d2-h5')
        path=verify_file(args.d2_h5,{'bytes':c['D2']['data_bytes'],'sha256':c['D2']['data_sha256']});cache=verify_file(B/'data_cache/d2_extracted_features.csv',{'bytes':(B/'data_cache/d2_extracted_features.csv').stat().st_size,'sha256':c['D2']['features_sha256']})
        raw=pd.read_hdf(path,'df');f=pd.read_csv(cache)
        if len(f)!=len(raw) or len(f)!=11136 or f.PCE_backward.notna().sum()!=5857:raise ValueError('D2 original counts differ')
        for name in ['date','patchID']:
            if not np.array_equal(raw[name].to_numpy(int),f[name].to_numpy(int)):raise ValueError('Raw/cache identity mismatch '+name)
        if not np.array_equal(raw.expID.to_numpy(str),f.expID.to_numpy(str)) or not np.allclose(raw.PCE_backward,f.PCE_backward,equal_nan=True):raise ValueError('Raw/cache experiment or target mismatch')
        if f.duplicated(['date','expID','patchID']).any() or f.groupby('expID').date.nunique().max()!=1:raise ValueError('D2 duplicate/reused experiment identity')
        # Preserve the actual published P2 cache rounding used by P6; audit it against every raw channel.
        diffs={}
        for ch in p.p4.CHANNELS:
            arrays=raw[ch].map(lambda x:np.asarray(x,float))
            if not arrays.map(lambda a:len(a)>0 and np.isfinite(a).all()).all():raise ValueError('Invalid raw trace')
            values={'mean':arrays.map(lambda a:float(a.mean())),'last':arrays.map(lambda a:float(a[-1])),'slope':arrays.map(lambda a:float(np.polyfit(np.linspace(0,1,len(a)),a,1)[0]))}
            for stat,v in values.items():
                name=ch+'_'+stat;delta=np.abs(v.to_numpy()-f[name].to_numpy());diffs[name]=float(delta.max())
                if not np.allclose(v,f[name],rtol=1e-10,atol=1e-7):raise ValueError('Raw/cache numeric mismatch '+name)
        meta=pd.DataFrame({'sample_id':f.sample_id,'unit_id':f.date.astype(str)+'::'+f.expID,'split_group':f.date.astype(str),'patchID':f.patchID.astype(int)})
        dates=set(map(str,c['D2']['development_dates']));hist=set(map(str,c['D2']['history_dates']));meta['role']=np.where(meta.split_group.isin(dates),'development',np.where(meta.split_group.isin(hist),'history','unlabeled_other_date'));lookup={str(date):i for i,dates_ in enumerate(c['D2']['outer_validation_dates']) for date in dates_};meta['outer_fold']=meta.split_group.map(lookup).fillna(-1).astype(int)
        y=meta.copy();y['target']=f.PCE_backward.to_numpy();counts=y.groupby('unit_id').target.transform('count');y=y[y.target.notna()&(counts>=4)&y.role.isin(['development','history'])].reset_index(drop=True)
        dev=y[y.role=='development'];histor=y[y.role=='history']
        if (len(dev),len(histor),dev.unit_id.nunique(),histor.unit_id.nunique(),dev.split_group.nunique(),histor.split_group.nunique())!=(4707,1150,158,36,14,3):raise ValueError('Frozen D2 cohort differs')
        d=out/'D2';d.mkdir();save_frame(meta,d/'optical_identity.csv');save_frame(y,d/'labels.csv');np.savez_compressed(d/'features.npz',FULL=f[p.p4.F2].to_numpy(float),RULES=f[p.p4.RULES].to_numpy(float))
        receipt['raw_inputs']['D2']={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)};receipt['datasets']['D2']={'development_rows':4707,'history_rows':1150,'development_groups':14,'history_groups':3,'development_experiments':158,'history_experiments':36,'feature_names':p.p4.F2,'rule_names':p.p4.RULES,'schema_columns':list(raw.columns),'label_zeros':int((y.target==0).sum()),'cache_sha256':sha(cache),'raw_F2_max_abs_differences':diffs}
    receipt['wall_seconds']=time.perf_counter()-start;receipt['files']={x.relative_to(out).as_posix():sha(x) for x in out.rglob('*') if x.is_file()};dump(out/'data_manifest.json',receipt);print('Prepared real frozen inputs, zero model fits:',out,flush=True)
def parser():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--dataset',choices=['D1','D2','both'],default='both');a.add_argument('--d1-train',type=Path);a.add_argument('--d1-test',type=Path);a.add_argument('--d2-h5',type=Path);a.add_argument('--output',type=Path,required=True);return a
if __name__=='__main__':prepare(parser().parse_args())
