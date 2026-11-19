"""Explicit replay entrypoints for the archived P7/P8/P9 supplementary procedures."""
from pathlib import Path
import argparse,shutil,sys,time
from release_common import B,read_json,dump,fresh_output,sha
STAGES={'P6':'pl_public_data_pilot_06_baselines_ablations','P7':'pl_public_data_pilot_07_robust_ranking','P8':'pl_public_data_pilot_08_reliability_gated_ranking','P9':'pl_public_data_pilot_09_trajectory_ranking'}
def setup(args):
    out=fresh_output(args.output);root=out/'workspace';name=STAGES[args.stage];stage=root/'research'/name
    shutil.copytree(B/'frozen_sources'/args.stage,stage);io_patches=[]
    if args.stage=='P9':
        # Numba's generated index filenames can exceed Windows MAX_PATH inside
        # the staged historical layout. Relocate only its disposable cache.
        import ast
        cache=B/'.numba_cache';cache.mkdir(exist_ok=True)
        for filename,old in [('r0_core.py',"str(BASE/'.numba_cache')"),('r1_core.py',"str(R1/'.numba_cache')")]:
            path=stage/'src'/filename;before=path.read_text(encoding='utf-8');after=before.replace(old,repr(str(cache)))
            funcs=lambda s:[ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            assert funcs(before)==funcs(after) and old in before
            path.write_text(after,encoding='utf-8');io_patches.append({'file':filename,'change':'Numba disposable cache path only','all_function_class_AST_unchanged':True})
    for ds in ['P6','P7']:
        source=B/'reference_results'/ds
        if source.exists():shutil.copytree(source,root/'research'/STAGES[ds]/'results',dirs_exist_ok=True)
    if args.stage=='P9':shutil.copytree(B/'reference_results/P9',stage/'results',dirs_exist_ok=True)
    sys.path.insert(0,str(stage/'src'))
    # Loader replacement changes IO only. All frozen model, splitting, candidate,
    # selection, objective and double-OOF functions stay byte-preserved.
    from dataset import Dataset
    import p6_core as core
    data={ds:Dataset(args.prepared,ds) for ds in ['D1','D2']}
    for obj in data.values():obj.validate_cohort()
    if args.stage in ['P7','P8']:
        import run_p6 as oldrun
        oldrun.read_d1=lambda _:data['D1'].legacy()
        oldrun.read_d2=lambda _:data['D2'].legacy()
        runner=__import__('run_'+args.stage.lower());run=lambda:runner.run(stage/'runs/reproduction')
    else:
        import pandas as pd
        import r0_core as r
        def development(audit_raw=False):
            result={}
            for ds,obj in data.items():
                dev=obj.role_groups('development');leg=obj.legacy();opt=obj.raw(dev) if ds=='D1' else None
                if ds=='D2':
                    spec=obj.manifest['raw_inputs']['D2'];path=Path(spec['path'])
                    if sha(path)!=spec['sha256']:raise ValueError('D2 raw source changed since preparation')
                    raw=pd.read_hdf(path,'df');opt=obj.meta[core.KEYS].copy()
                    for ch in r.CHANNELS:opt[ch]=raw[ch].to_numpy()
                    rawopt=opt.copy();opt=opt[opt.split_group.isin(dev)].reset_index(drop=True)
                else:rawopt=opt
                result[ds]={'opt':opt,'y':leg['y'],'features':obj.optical(groups=dev),'folds':leg['folds'],'raw_opt':rawopt,'schema':obj.spec['schema_columns'],'history_identity':obj.meta.loc[obj.meta.role=='history',core.KEYS].reset_index(drop=True)}
            return result
        r.load_development=development
        import run_r1 as runner
        run=lambda:runner.run(stage/'R1/runs/reproduction')
    receipt={'stage':args.stage,'workspace':str(root),'frozen_source_copied_byte_for_byte':not bool(io_patches),'runtime_IO_patches':io_patches,'model_functions_and_parameters_unchanged':True,'loader_adaptation':'prepared exact inputs and raw D2 optical arrays; targets supplied only through original supervised training/after-score evaluation boundaries','other_stages_started':False,'dry_run':args.dry_run,'new_model_fits':0 if args.dry_run else 'See original stage run_receipt.json','prepared_input_manifest_sha256':sha(Path(args.prepared)/'data_manifest.json')};dump(out/'replay_setup.json',receipt)
    return out,run
def main(args):
    out,run=setup(args)
    if args.dry_run:print(args.stage,'imports and runtime IO resolved; no training invoked',flush=True);return
    start=time.perf_counter()
    try:run()
    finally:dump(out/'replay_wall.json',{'stage':args.stage,'wall_seconds':time.perf_counter()-start,'parameters_changed':False,'historical_failure_statuses_retained':True})
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--stage',choices=['P7','P8','P9'],required=True);a.add_argument('--prepared',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--dry-run',action='store_true');main(a.parse_args())
