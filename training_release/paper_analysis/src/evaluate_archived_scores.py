"""Reproduce reported group metrics from frozen predictions, without fitting models."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
B=Path(__file__).resolve().parents[1]
def read(n,**kw):return pd.read_csv(B/n,float_precision='round_trip',**kw)
def evaluate():
    out=B/'REPRODUCED';out.mkdir(exist_ok=True);units=[];checks=[]
    for ds,phase,file in [('D1','development','d1_train_oof_predictions.csv'),('D1','historical','d1_author_test_predictions.csv'),('D2','development','d2_dev_oof_predictions.csv'),('D2','historical','d2_history_predictions.csv')]:
        q=read('SOURCE_DATA/P6/'+file,dtype={'split_group':str});assert q.sample_id.is_unique
        if ds=='D1':
            q=q.merge(read(f'SOURCE_DATA/P6/d1_{phase}_targets_full_precision.csv'),on='sample_id',validate='one_to_one');q['target']=q.raw_target
        assert q.groupby('split_group').outer_fold.nunique().eq(1).all()
        methods=[m for m in ['PAIR_ABS','PL_INTENSITY','PL_PAIRWISE','RIDGE_PL','ID_PAIRWISE','ID_PLUS_PL_PAIRWISE','LGBM_PL','EXTRATREES_PL','HISTGB_PL','AUTHOR_KNN_AUX','F2_RIDGE_AUX'] if m in q]
        for m in methods:
            assert np.isfinite(q[m]).all()
            for uid,g in q.groupby('unit_id'):
                k=int(np.ceil(len(g)/4));s=g.sort_values([m,'sample_id'],ascending=[False,True],kind='stable').head(k);t=g.sort_values(['target','sample_id'],ascending=[False,True],kind='stable').head(k)
                rho=float(spearmanr(g[m],g.target).statistic) if g[m].nunique()>1 and g.target.nunique()>1 else np.nan
                units.append(dict(dataset=ds,phase=phase,model_id=m,unit_id=uid,split_group=str(g.split_group.iloc[0]),outer_fold=int(g.outer_fold.iloc[0]),n=len(g),k=k,top25_gain_pp=s.target.mean()-g.target.mean(),rho=rho,top25_overlap=len(set(s.sample_id)&set(t.sample_id))/k))
    u=pd.DataFrame(units);cols=['top25_gain_pp','rho','top25_overlap'];ind=u.groupby(['dataset','phase','model_id','split_group'])[cols].mean().reset_index();macro=ind.groupby(['dataset','phase','model_id'])[cols].mean().reset_index()
    expected=read('ANALYSIS/recomputed_macro_metrics.csv')
    for _,r in macro.iterrows():
        e=expected[(expected.dataset==r.dataset)&(expected.phase==r.phase)&(expected.model_id==r.model_id)].iloc[0];delta=max(abs(r[c]-e[c]) for c in cols);assert delta<1e-8
        checks.append(dict(dataset=r.dataset,phase=r.phase,model=r.model_id,max_error=delta))
    for name,frame in [('selection_unit_metrics',u),('independent_group_metrics',ind),('macro_metrics',macro)]:frame.to_csv(out/(name+'.csv'),index=False,float_format='%.17g')
    (out/'metric_checks.json').write_text(json.dumps(dict(model_scopes=len(checks),checks=checks,new_model_fits=0,passed=True),indent=2),encoding='utf-8');print('PASS',len(checks),'archived model-scope metric comparisons; zero model fits')
if __name__=='__main__':evaluate()
