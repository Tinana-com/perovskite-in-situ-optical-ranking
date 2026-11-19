"""Rebuild the original P2 optical cache from the genuine D2 H5, without fitting models."""
import argparse,sys,time
from pathlib import Path
import pandas as pd
from release_common import B,sha,read_json,dump,fresh_output
from prepare_data import verify_file
from portable_models import named_source
def run(args):
    c=read_json(B/'configs/frozen_p6_protocol.json')['D2'];path=verify_file(args.d2_h5,{'bytes':c['data_bytes'],'sha256':c['data_sha256']});out=fresh_output(args.output);p=named_source('frozen_p2',B/'frozen_sources/P2/src/run_p2.py');start=time.perf_counter();raw=pd.read_hdf(path,'df');raw['date']=raw.date.astype(int);meta=raw[['date','expID','patchID','PCE_backward','ratio','molarity','evac_duration']].copy();meta['sample_id']=[f'D2S_{i+1:05d}' for i in range(len(raw))];meta['sequence_length']=raw.ND.map(len).to_numpy();features=p.build_features(raw);q=pd.concat([meta,features.drop(columns=['ratio','molarity','evac_duration'])],axis=1);dest=out/'extracted_features.csv';p.write_csv(q,dest)
    expected=pd.read_csv(B/'data_cache/d2_extracted_features.csv');got=pd.read_csv(dest)
    import numpy as np
    assert got.columns.tolist()==expected.columns.tolist() and got.shape==expected.shape
    assert got[['sample_id','date','expID','patchID']].equals(expected[['sample_id','date','expID','patchID']])
    names=got.select_dtypes(include='number').columns;diff={n:float(np.nanmax(abs(got[n]-expected[n]))) if got[n].notna().any() else 0. for n in names};numeric=all(np.allclose(got[n],expected[n],rtol=1e-10,atol=1e-7,equal_nan=True) for n in names)
    dump(out/'cache_rebuild_receipt.json',{'rows':len(got),'columns':len(got.columns),'source_sha256':sha(path),'generated_sha256':sha(dest),'frozen_cache_sha256':c['features_sha256'],'byte_identical':sha(dest)==c['features_sha256'],'numeric_equivalent':numeric,'max_abs_differences':diff,'model_fits':0,'wall_seconds':time.perf_counter()-start});assert numeric;print('Raw D2 cache rebuilt; numeric match:',numeric,'byte match:',sha(dest)==c['features_sha256'],flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--d2-h5',type=Path,required=True);a.add_argument('--output',type=Path,required=True);run(a.parse_args())
