"""Interface/identity tests use real excerpts and mocked supervised estimators."""
from pathlib import Path
import sys
import numpy as np,pandas as pd,pytest
from sklearn.base import BaseEstimator
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from dataset import Dataset
from release_common import load_core,fresh_output,read_json
import portable_models as pm
import train
@pytest.fixture(params=['D1','D2'])
def data(request):return Dataset(B/'tests/fixtures',request.param)
@pytest.fixture
def core():return load_core()
def test_fixture_hash_and_identity(data):
    assert data.meta.sample_id.is_unique
    assert set(data.labels.sample_id)<=set(data.meta.sample_id)
    assert data.meta.groupby('unit_id').split_group.nunique().eq(1).all()
def test_incomplete_fixture_cannot_train_as_full_cohort(data):
    with pytest.raises(ValueError,match='cohort'):data.validate_cohort()
def test_train_validation_groups_disjoint(data):
    for _,tr,ev in data.cases():
        y,keys=train.build_case(data,tr,ev);assert not set(y.split_group)&set(keys.split_group);assert not set(y.unit_id)&set(keys.unit_id)
def test_every_development_id_has_one_oof_group(data):
    groups=[group for _,_,ev in data.cases() for group in ev]
    assert len(groups)==len(set(groups)) and set(groups)==data.role_groups('development')
@pytest.mark.parametrize('action',['zero','shuffle','mask'])
def test_validation_label_interventions_leave_training_unchanged(data,action):
    _,tr,ev=next(data.cases());before,ids=train.build_case(data,tr,ev);part=data.labels.split_group.isin(ev)
    data.labels.loc[part,'target']=0 if action=='zero' else np.nan if action=='mask' else np.random.default_rng(41).permutation(data.labels.loc[part,'target'])
    after,newids=train.build_case(data,tr,ev);pd.testing.assert_frame_equal(before,after);pd.testing.assert_frame_equal(ids,newids)
def test_optical_prediction_input_excludes_labels(data):
    assert not {'target','pce_mean','PCE_backward','outer_fold','role'}&set(data.optical().columns)
def test_predictor_does_not_need_label_file(data,tmp_path):
    import shutil
    out=tmp_path/'nolabel';shutil.copytree(B/'tests/fixtures',out);(out/data.name/'labels.csv').unlink();q=Dataset(out,data.name,load_labels=False);assert q.labels.empty and not q.optical().empty
@pytest.mark.parametrize('window',['FULL','EARLY25','LATE25'])
def test_d1_features_equal_original_raw_extraction(window,core):
    d=Dataset(B/'tests/fixtures','D1');expected=core.d1_features(d.raw(),window);np.testing.assert_array_equal(d.optical(window)[core.p5.STATS],expected[core.p5.STATS])
def test_d1_nonfinite_window_repair_retains_shape(core):
    d=Dataset(B/'tests/fixtures','D1');a=d.traces[0].copy();a[5:9]=np.nan;v=core.clean_window(a,'FULL');assert v.shape==(719,) and np.isfinite(v).all()
def test_d1_constant_and_all_nan_are_finite(core):
    assert np.array_equal(core.clean_window(np.full(719,np.nan),'FULL'),np.zeros(719))
    assert np.isfinite(core.clean_window(np.ones(719),'EARLY25')).all()
def test_relative_transform_training_floor_independent_of_eval_targets(data,core):
    _,tr,ev=next(data.cases());spec=next(s for s in pm.configuration(data.name) if s['model_id']=='PL_PAIRWISE')
    if spec['family']=='INHERITED_CONTROL':spec={'position':'PL','representation':'REL','statistics_subset':'ALL'}
    pre=core.Preprocessor(data.name,spec,read_json(B/'configs/frozen_p6_protocol.json')).fit(data.optical(groups=tr));before=pre.transform(data.optical(groups=ev));data.labels.loc[data.labels.split_group.isin(ev),'target']=1e6;np.testing.assert_array_equal(before,pre.transform(data.optical(groups=ev)))
def test_frozen_pair_tolerance_differs_between_datasets(core):
    # Boundary is a deterministic unit fixture, not generated experimental data.
    y=pd.DataFrame([['a','u','d',11,0.],['b','u','d',12,.5]],columns=core.KEYS+['target']);x=np.array([[0.],[1.]])
    with pytest.raises(ValueError,match='informative'):core.pairs(y,x,'D1')
    y['split_group']='20230425';assert len(core.pairs(y,x,'D2')[1])==2
def test_pair_weights_and_symmetry_on_real_excerpts(data,core):
    _,tr,_=next(data.cases());y=data.y(tr).sort_values(['split_group','unit_id','sample_id']).reset_index(drop=True);f=data.optical(groups=tr).set_index('sample_id');x=f.loc[y.sample_id,data.spec['feature_names']].to_numpy();xx,labels,weights,audit,ids=core.pairs(y,x,data.name)
    assert len(xx)==len(labels)==len(weights) and set(labels)=={0,1} and np.all(weights>0) and abs(weights.mean()-1)<1e-12 and len(ids)*2==len(xx)
def test_top25_rounding_and_stable_ties(core):
    assert core.top_k(9)==3
    assert core.order([1,1,0],['b','a','c']).tolist()==[1,0,2]
def test_constant_spearman_is_undefined(core):assert np.isnan(core.rho([1,2,3],[1,1,1]))
def test_lambda_queries_are_training_unit_contiguous(data,core):
    y=data.y(data.role_groups('development')).sort_values('sample_id').reset_index(drop=True);x=np.zeros((len(y),2));_,relevance,sizes,seq,q=core.rank_queries(y,x);assert sizes.sum()==len(y) and relevance.min()>=0 and relevance.max()<=4 and q.groupby('unit_id').apply(lambda g:g.index.max()-g.index.min()+1==len(g),include_groups=False).all()
def test_output_does_not_overwrite_previous_attempt(tmp_path):
    p=tmp_path/'attempt';fresh_output(p);(p/'result').write_text('preserve')
    with pytest.raises(FileExistsError):fresh_output(p)
def test_scientific_source_directories_are_not_output_targets():
    with pytest.raises(ValueError):fresh_output(B/'frozen_sources/P6/new_attempt')
class SpyEstimator(BaseEstimator):
    """Shape-only test double: no statistical estimator is trained."""
    def __init__(self,**kw):self.kw=kw;self.n_iter_=np.array([1])
    def fit(self,x,y,**kwargs):
        self.x=np.asarray(x);self.y=np.asarray(y);assert len(self.x)==len(self.y) and np.isfinite(self.x).all();return self
    def decision_function(self,x):return np.asarray(x)[:,0]
    def predict(self,x):return np.asarray(x)[:,0]
@pytest.mark.parametrize('family',['PAIRWISE','RIDGE','EXTRATREES','HISTGB','LGBM'])
def test_native_families_route_frozen_arguments_with_mocked_fit(data,core,monkeypatch,family):
    for name in ['LogisticRegression','Ridge','ExtraTreesRegressor','HistGradientBoostingRegressor','LGBMRanker']:monkeypatch.setattr(core,name,SpyEstimator)
    spec=next(s for s in pm.configuration(data.name) if s['family']==family);_,tr,ev=next(data.cases());model=pm.fit(data,spec,tr);pred=model.predict(data,ev)
    assert len(pred)==len(data.optical(groups=ev)) and 'target' not in pred
    expected=read_json(B/'configs/frozen_p6_protocol.json')['models'][family]
    if family=='PAIRWISE':expected={'random_state':20261019}|expected
    assert model.model.model_.kw==expected
def test_registry_has_all_trained_p6_and_inherited_controls(core):
    assert len(pm.configuration('D1'))==20 and len(pm.configuration('D2'))==14
    assert {q['model_id'] for q in pm.configuration('D2')}>= {'PAIR_ABS','F2_RIDGE_AUX','PL_PAIRWISE','ID_PAIRWISE','ID_PLUS_PL_PAIRWISE','ID_PRIOR','PL_INTENSITY'}
def test_main_full_plan_freezes_development_before_history(data):
    plan=train.plan(data,pm.configuration(data.name),'full');assert plan[-1]['phase']=='history' and all(x['phase']=='development' for x in plan[:-1]);assert not set(plan[-1]['training_groups'])&set(plan[-1]['validation_groups'])
def test_rule_inference_label_changes_do_not_change_predictions():
    data=Dataset(B/'tests/fixtures','D1');model=pm.Fitted('D1','PL_INTENSITY','D1_RULE',None,rule=('mean',1));groups=data.role_groups('development');before=model.predict(data,groups);data.labels['target']=np.nan;pd.testing.assert_frame_equal(before,model.predict(data,groups))

class SpyPrior:
    def __init__(self,*a,**kw):pass
    def fit(self,y):
        assert 'pce_mean' in y or 'PCE_backward' in y
        self.n_=len(y);return self
    def predict(self,ids):
        assert not {'target','pce_mean','PCE_backward'}&set(ids.columns)
        return ids.patchID.to_numpy(float)
@pytest.mark.parametrize('ds,mid',[
 ('D1','ID_PRIOR'),('D1','PL_INTENSITY'),('D1','AUTHOR_KNN_AUX'),
 ('D2','ID_PRIOR'),('D2','PL_INTENSITY'),('D2','F2_RIDGE_AUX'),
 ('D2','ID_PAIRWISE'),('D2','PL_PAIRWISE'),('D2','ID_PLUS_PL_PAIRWISE')])
def test_all_inherited_control_training_adapters_with_mocked_estimators(ds,mid,monkeypatch):
    import sklearn.linear_model,sklearn.neighbors
    p,position,p2=pm.dependencies();data=Dataset(B/'tests/fixtures',ds);_,tr,ev=next(data.cases())
    monkeypatch.setattr(p.p5,'IDPrior',SpyPrior);monkeypatch.setattr(position,'IDPrior',SpyPrior)
    monkeypatch.setattr(sklearn.linear_model,'LogisticRegression',SpyEstimator)
    monkeypatch.setattr(sklearn.neighbors,'KNeighborsRegressor',SpyEstimator)
    monkeypatch.setattr(p2,'Ridge',SpyEstimator)
    monkeypatch.setattr(p.p5,'choose_knn',lambda y,opt,folds,c:(14,pd.DataFrame()))
    spec=next(s for s in pm.configuration(ds) if s['model_id']==mid)
    fitted=pm.fit(data,spec,tr);scores=fitted.predict(data,ev)
    assert np.isfinite(scores.score).all() and 'target' not in scores
    assert set(fitted.info['training_groups'])==tr
