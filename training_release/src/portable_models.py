"""Frozen P6 models plus faithful training adapters for its inherited controls."""
import sys,importlib.util,hashlib
from dataclasses import dataclass
from release_common import B,load_core,read_json
import numpy as np,pandas as pd
def named_source(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def dependencies():
    p=load_core();position=named_source('frozen_position',B/'frozen_sources/POSITION/src/run_position_audit.py');p2=named_source('frozen_p2',B/'frozen_sources/P2/src/run_p2.py');return p,position,p2
def as_d1(f):
    q=f.copy();q['substrateID']=q.unit_id;return q.drop(columns=['unit_id','split_group'])
def as_d2(f):
    q=f.copy();q['date']=q.split_group.astype(int);q['expID']=q.unit_id.str.split('::',n=1).str[1];return q.drop(columns=['unit_id','split_group'])
def configuration(ds):
    p=load_core();grid=[s for s in p.grid() if s['dataset']==ds];ids={s['model_id'] for s in grid}
    auxiliary=['ID_PRIOR','PL_INTENSITY','AUTHOR_KNN_AUX'] if ds=='D1' else ['ID_PRIOR','PL_INTENSITY','F2_RIDGE_AUX','ID_PAIRWISE','PL_PAIRWISE','ID_PLUS_PL_PAIRWISE']
    return grid+[dict(dataset=ds,model_id=mid,family='INHERITED_CONTROL',window='FULL',role='MAIN',status='RUN',new_fit=True) for mid in auxiliary if mid not in ids]
@dataclass
class Fitted:
    dataset:str
    model_id:str
    kind:str
    model:object
    window:str='FULL'
    rule:tuple|str|None=None
    info:dict|None=None
    def predict(self,data,groups):
        p,_,_=dependencies();f=data.optical(self.window,groups)
        if self.kind=='P6':score=self.model.predict(f)
        elif self.kind=='D1_PRIOR':score=self.model.predict(as_d1(f[p.KEYS]))
        elif self.kind=='D2_PRIOR':score=self.model.predict(as_d2(f[p.KEYS]))
        elif self.kind=='D1_RULE':score=self.rule[1]*f[self.rule[0]].to_numpy()
        elif self.kind=='D2_RULE':
            f=data.rules(groups);score=f[self.rule].to_numpy()
        elif self.kind=='D1_KNN':score=self.model.predict(p.p5.raw_signals(as_d1(data.raw(groups))))
        elif self.kind in ['D2_ID_PAIR','D2_IDPL_PAIR','D2_PL_PAIR','D2_F2_RIDGE']:
            inp=as_d2(f);score=self.model.predict(inp[p.p4.KEYS] if self.kind=='D2_ID_PAIR' else inp)
        else:raise ValueError('Unsupported artifact kind '+self.kind)
        q=f[p.KEYS].copy();q['score']=np.asarray(score,float)
        if not np.isfinite(q.score).all() or q.sample_id.duplicated().any():raise ValueError('Nonfinite/duplicated predictions')
        return q
def fit(data,spec,train_groups):
    """This is the explicit supervised entrypoint; caller supplies TRAIN groups only."""
    p,position,p2=dependencies();ds=data.name;mid=spec['model_id'];window=spec['window'];y=data.y(train_groups);f=data.optical(window,train_groups);info={'training_groups':sorted(train_groups),'training_labeled_rows':len(y),'training_optical_rows':len(f),'seed':20261019,'model_id':mid}
    if set(y.split_group)!=set(train_groups) or not set(y.sample_id)<=set(f.sample_id):raise ValueError('Training group/identity mismatch')
    if spec['family']!='INHERITED_CONTROL':
        model=p.Model(ds,spec,read_json(B/'configs/frozen_p6_protocol.json')).fit(y,f);kind='P6'
        info.update(n_features=len(model.pre_.names_),pair_ID_sha256=getattr(model,'pair_hash_',None),ordered_pair_count=getattr(model,'n_pairs_',0))
        if model.pair_audit_ is not None:info['pair_audit']=model.pair_audit_.to_dict('records')
        return Fitted(ds,mid,kind,model,window,info=info)
    if ds=='D1':
        c=read_json(B/'configs/frozen_p5_d1_protocol.json');yf=as_d1(y).rename(columns={'target':'pce_mean'});ff=as_d1(f);opt=as_d1(data.raw(train_groups));folds=data.labels[data.labels.split_group.isin(train_groups)][p.KEYS+['outer_fold']].rename(columns={'outer_fold':'foldNo'});folds=as_d1(folds)
        if mid=='ID_PRIOR':model=p.p5.IDPrior().fit(yf);kind='D1_PRIOR'
        elif mid=='PL_INTENSITY':
            rule,rid,detail=p.p5.choose_rule(yf,ff,folds,c);info.update(selected_rule=rid,inner_metrics=detail.to_dict('records'));return Fitted(ds,mid,'D1_RULE',None,rule=rule,info=info)
        elif mid=='AUTHOR_KNN_AUX':
            from sklearn.neighbors import KNeighborsRegressor
            k,detail=p.p5.choose_knn(yf,opt,folds,c);target=yf.set_index('sample_id').loc[opt.sample_id].pce_mean.to_numpy();model=KNeighborsRegressor(n_neighbors=k,weights='uniform',n_jobs=1,algorithm='brute').fit(p.p5.raw_signals(opt),target);kind='D1_KNN';info.update(selected_k=k,inner_metrics=detail.to_dict('records'))
        else:raise ValueError(mid)
    else:
        c=read_json(B/'configs/frozen_position_protocol.json');yf=as_d2(y).rename(columns={'target':'PCE_backward'});ff=as_d2(f)
        if mid=='ID_PRIOR':model=position.IDPrior(5.).fit(yf);kind='D2_PRIOR'
        elif mid=='PL_INTENSITY':
            rules=as_d2(data.rules(train_groups));tr=yf.merge(rules,on=p.p4.KEYS,validate='one_to_one');pc=read_json(B/'frozen_sources/P4/configs/frozen_p4_protocol.json');rule,detail,ranking=p.p4.select_naive(tr,pc,sorted(map(int,train_groups)),0);info.update(selected_rule=rule,inner_ranking=ranking.to_dict('records'));return Fitted(ds,mid,'D2_RULE',None,rule=rule,info=info)
        elif mid in ['ID_PAIRWISE','ID_PLUS_PL_PAIRWISE']:
            model=position.ControlledPairwise(c,include_PL=mid=='ID_PLUS_PL_PAIRWISE').fit(yf,ff);kind='D2_IDPL_PAIR' if mid=='ID_PLUS_PL_PAIRWISE' else 'D2_ID_PAIR';info.update(ordered_pair_count=model.n_pairs_,pair_ID_sha256=hashlib.sha256('\n'.join(model.pair_ids_).encode()).hexdigest())
        elif mid=='PL_PAIRWISE':
            pc=read_json(B/'frozen_sources/P4/configs/frozen_p4_protocol.json');model=p.p4.PairRanker(pc).fit(yf,ff);kind='D2_PL_PAIR';info.update(ordered_pair_count=model.n_pairs_,pair_ID_sha256=hashlib.sha256('\n'.join(model.pair_ids_).encode()).hexdigest())
        elif mid=='F2_RIDGE_AUX':
            pc=read_json(B/'frozen_sources/P2/configs/frozen_p2_protocol.json');model=p2.make_model({'family':'F2','model':'ridge'},pc);train=ff.set_index('sample_id').loc[yf.sample_id].reset_index();model.fit(train,yf.PCE_backward.to_numpy(float));kind='D2_F2_RIDGE'
        else:raise ValueError(mid)
    return Fitted(ds,mid,kind,model,window,info=info)
