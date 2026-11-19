"""Publication figures from archived outputs; no invented experimental values."""
import os,sys
from pathlib import Path
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(B/'REPRODUCED/.mplconfig')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'savefig.facecolor':'white'})
BLUE='#176B92';TEAL='#00857D';ORANGE='#D17A36';RED='#B74149';GREY='#667780';LIGHT='#EDF3F6';DARK='#18364B'
def read(p,**kw):return pd.read_csv(B/p,float_precision='round_trip',**kw)
M=read('ANALYSIS/recomputed_macro_metrics.csv');U=read('ANALYSIS/recomputed_independent_group_metrics.csv',dtype={'split_group':str});P=read('ANALYSIS/paired_gain_comparisons.csv')
PROVENANCE=[]
def finish(fig,name,sources,description):
    dest=B/'FIGURES';dest.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.09)
    plt.close(fig);PROVENANCE.append({'figure':name,'source_files':sources,'content':description})
def panel(ax,letter,title):ax.set_title(letter+'  '+title,loc='left',fontweight='bold',pad=9)
def horizontal(ax):ax.xaxis.grid(True,color='#DDE5E9',linewidth=.5);ax.set_axisbelow(True)
def box(ax,xy,w,h,text,color=LIGHT,fs=10):
    p=FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.012,rounding_size=.018',facecolor=color,edgecolor='#B6C7D0',linewidth=.8);ax.add_patch(p);ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=fs,color=DARK)
def arrow(ax,start,end):ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,linewidth=1,color=GREY))
def f2():
    fig,axes=plt.subplots(2,2,figsize=(7.5,5.2),gridspec_kw={'height_ratios':[1,1.2]});cnt=read('ANALYSIS/dataset_counts.csv')
    for j,ds in enumerate(['D1','D2']):
        ax=axes[0,j];g=cnt[cnt.dataset==ds];v=[g[g.phase==p].regions.iloc[0] for p in ['development','historical']];bars=ax.bar(['Development','Historical'],v,color=[BLUE,ORANGE],width=.55)
        for bar,n in zip(bars,v):ax.text(bar.get_x()+bar.get_width()/2,n+max(v)*.04,f'{n:,}',ha='center',fontsize=9)
        ax.set_ylim(0,max(v)*1.26);ax.set_ylabel('Labeled device regions');panel(ax,chr(97+j),ds+' archived evaluation cohorts')
    groups=read('ANALYSIS/group_characteristics.csv',dtype={'split_group':str});g=groups[(groups.dataset=='D1')&(groups.phase=='development')].sort_values('split_group');a=axes[1,0];a.imshow([g.outer_fold.to_numpy()],aspect='auto',cmap='Blues',vmin=0,vmax=6);a.set_yticks([0],['Author fold']);a.set_xticks(range(len(g)),g.split_group,rotation=90,fontsize=6.4);a.set_xlabel('26 substrates; each is one evaluation group');panel(a,'c','D1 substrate isolation')
    for x,fold in enumerate(g.outer_fold):a.text(x,0,str(fold),ha='center',va='center',fontsize=6.5,color='white' if fold>=3 else DARK)
    a=axes[1,1];g=groups[(groups.dataset=='D2')&(groups.phase=='development')][['split_group','outer_fold']].drop_duplicates().sort_values('split_group');a.imshow([g.outer_fold.to_numpy()],aspect='auto',cmap='Blues',vmin=0,vmax=6);a.set_yticks([0],['Date fold']);a.set_xticks(range(len(g)),[d[4:6]+'/'+d[6:] for d in g.split_group],rotation=65,fontsize=7);a.set_xlabel('14 dates; experiments remain within each date');panel(a,'d','D2 date isolation')
    for x,fold in enumerate(g.outer_fold):a.text(x,0,str(fold),ha='center',va='center',fontsize=7.5,color='white' if fold>=3 else DARK)
    for a in axes[1]:
        for s in a.spines.values():s.set_visible(False)
        a.tick_params(axis='both',length=0)
    fig.tight_layout(h_pad=2.2,w_pad=2)
    finish(fig,'Figure_2_data_protocol',['ANALYSIS/dataset_counts.csv','ANALYSIS/group_characteristics.csv'],'Exact archived labeled counts and original five/seven outer-fold memberships; historical partitions reported separately.')
LABELS={'PAIR_ABS':'Pairwise absolute','PL_PAIRWISE':'Pairwise reference','ID_PLUS_PL_PAIRWISE':'Position + optical','LGBM_PL':'LambdaRank','EXTRATREES_PL':'ExtraTrees','HISTGB_PL':'Gradient boosting','RIDGE_PL':'Ridge','ID_PAIRWISE':'Position pairwise','ID_PRIOR':'Position prior','PL_INTENSITY':'PL intensity rule','AUTHOR_KNN_AUX':'Author kNN reference','F2_RIDGE_AUX':'Absolute Ridge reference'}
ORDER=['PAIR_ABS','PL_PAIRWISE','ID_PLUS_PL_PAIRWISE','LGBM_PL','EXTRATREES_PL','HISTGB_PL','RIDGE_PL','ID_PAIRWISE','ID_PRIOR','PL_INTENSITY']
def f4():
    fig,axes=plt.subplots(2,2,figsize=(7.5,5.8))
    ax=axes[0,0];names=['Absolute','Relative','Combined'];x=np.arange(3)
    for offset,ds,color,ids in [(-.16,'D1',BLUE,['PAIR_ABS','PAIR_REL','PL_PAIRWISE']),(.16,'D2',ORANGE,['PAIR_ABS','PL_PAIRWISE','PAIR_BOTH'])]:
        q=M[(M.dataset==ds)&(M.phase=='development')].set_index('model_id').loc[ids];ax.bar(x+offset,q.top25_gain_pp,width=.3,color=color,label=ds)
    ax.set_xticks(x,names);ax.set_ylabel('Top25 PCE gain (pp)');ax.legend(frameon=False);panel(ax,'a','Feature representation')
    for ax,a,b,title,letter in [(axes[0,1],'ID_PLUS_PL_PAIRWISE','ID_PAIRWISE','Optical increment beyond position','b'),(axes[1,0],'PL_PAIRWISE','RIDGE_PL','Pairwise versus point regression','c')]:
        q=P[(P.phase=='development')&(P.method_a==a)&(P.method_b==b)].set_index('dataset').loc[['D1','D2']];y=np.arange(2);ax.errorbar(q.delta_gain_pp,y,xerr=[q.delta_gain_pp-q.CI_low,q.CI_high-q.delta_gain_pp],fmt='o',color=TEAL,ecolor=GREY,capsize=3);ax.set_yticks(y,['D1','D2']);ax.axvline(0,color=GREY,lw=.7);ax.set_xlabel('Paired gain difference (pp)');horizontal(ax);panel(ax,letter,title)
    ax=axes[1,1];ids=['PAIR_EARLY25','PAIR_LATE25','PL_PAIRWISE','PAIR_LEVEL','PAIR_DYNAMIC'];q=M[(M.dataset=='D1')&(M.phase=='development')].set_index('model_id').loc[ids];ax.bar(range(5),q.top25_gain_pp,color=[GREY,GREY,BLUE,TEAL,ORANGE]);ax.set_xticks(range(5),['First25%','Last25%','Full','Level','Dynamic'],rotation=25,ha='right',fontsize=7.5);ax.set_ylabel('Top25 PCE gain (pp)');panel(ax,'d','D1 temporal and statistic ablations')
    fig.tight_layout(h_pad=2,w_pad=1.8)
    finish(fig,'Figure_4_comparisons_ablations',['ANALYSIS/recomputed_macro_metrics.csv','ANALYSIS/paired_gain_comparisons.csv','SOURCE_DATA/P6/d1_ablation_long.csv','SOURCE_DATA/P6/d2_ablation_long.csv'],'Exact like-feature objective and position contrasts; temporal/level/dynamic ablations use D1 combined representation. D2 temporal/dynamic not run, not filled with zero.')
def f5():
    fig,axes=plt.subplots(2,2,figsize=(7.6,6.1),gridspec_kw={'height_ratios':[1.15,1]})
    for j,ds in enumerate(['D1','D2']):
        ax=axes[0,j];q=U[(U.dataset==ds)&(U.phase=='development')&(U.model_id=='PAIR_ABS')].sort_values('split_group');x=np.arange(len(q));ax.bar(x,q.top25_gain_pp,color=[RED if v<0 else BLUE for v in q.top25_gain_pp],width=.72);ax.axhline(0,color=GREY,lw=.7);labels=q.split_group if ds=='D1' else [v[4:6]+'/'+v[6:] for v in q.split_group];ax.set_xticks(x,labels,rotation=90 if ds=='D1' else 65,fontsize=6.3 if ds=='D1' else 7);ax.set_ylabel('Top25 PCE gain (pp)');panel(ax,chr(97+j),ds+' all development groups')
    ax=axes[1,0];rows=[]
    for label,stage,method in [('Dual view','P7','DVR_SELECTED'),('Gated residual','P8','P8_SELECTED')]:
        data=read(f'SOURCE_DATA/{stage}/macro_model_comparison.csv');data=data[(data.model_id==method)&data.scope.str.contains('DEVELOPMENT')]
        for _,z in data.iterrows():base=M[(M.dataset==z.dataset)&(M.phase=='development')&(M.model_id=='PAIR_ABS')].top25_gain_pp.iloc[0];rows.append({'stage':label,'dataset':z.dataset,'delta':z.top25_gain_pp-base})
    p9=read('SOURCE_DATA/P9/dataset_macro_metrics.csv')
    for _,z in p9[p9.model_id=='M1'].iterrows():base=M[(M.dataset==z.dataset)&(M.phase=='development')&(M.model_id=='PAIR_ABS')].top25_gain_pp.iloc[0];rows.append({'stage':'Shape extension','dataset':z.dataset,'delta':z.top25_gain_pp-base})
    rr=pd.DataFrame(rows)
    for ds,color,offset in [('D1',BLUE,-.13),('D2',ORANGE,.13)]:
        q=rr[rr.dataset==ds];ax.scatter(q.delta,np.arange(len(q))+offset,label=ds,color=color,s=25)
    ax.set_yticks(range(3),['Dual view','Gated residual','Shape extension']);ax.axvline(0,color=GREY,lw=.7);ax.set_xlabel('Gain difference versus absolute pairwise (pp)');ax.legend(frameon=False,loc='lower left');horizontal(ax);panel(ax,'c','Archived model extensions')
    ax=axes[1,1];cost=read('SOURCE_DATA/P9/model_resource_summary.csv')
    for ds,color in [('D1',BLUE),('D2',ORANGE)]:
        q=cost[cost.dataset==ds];metrics=p9[p9.dataset==ds].set_index('model_id');xx=q.total_pipeline_wall_seconds;yy=metrics.loc[q.model_id].top25_gain_pp.to_numpy();ax.plot(xx,yy,color=color,lw=.7,alpha=.6);ax.scatter(xx,yy,color=color,s=26,label=ds)
        for x,y,mid in zip(xx,yy,q.model_id):
            off=(-20,-15) if ds=='D2' and mid=='M2' else (-3,12) if ds=='D2' and mid=='M3' else (3,4)
            ax.annotate(mid,(x,y),xytext=off,textcoords='offset points',fontsize=7,arrowprops={'arrowstyle':'-','lw':.5,'color':GREY} if ds=='D2' and mid in ['M2','M3'] else None)
        if ds=='D2':
            sub=q[q.model_id.isin(['M2','M3'])];ax.scatter(sub.total_pipeline_wall_seconds,metrics.loc[sub.model_id].top25_gain_pp,marker='x',color=RED,s=45,lw=1.4)
    ax.set_xscale('log');ax.set_xlabel('Total outer-fold pipeline time (s)');ax.set_ylabel('Top25 PCE gain (pp)');panel(ax,'d','P9 recorded cost and performance')
    fig.tight_layout(h_pad=2.2,w_pad=1.7)
    finish(fig,'Figure_5_groups_complexity',['ANALYSIS/recomputed_independent_group_metrics.csv','SOURCE_DATA/P7/macro_model_comparison.csv','SOURCE_DATA/P8/macro_model_comparison.csv','SOURCE_DATA/P9/dataset_macro_metrics.csv','SOURCE_DATA/P9/model_resource_summary.csv'],'Every development substrate/date retained; red crosses flag unconverged D2 MiniRocket branches, not confirmed optima. Runtime is recorded total outer-fold pipeline cost, not per-device deployment time.')
def supplemental():
    fig,axes=plt.subplots(1,4,figsize=(12.5,6.5),sharex=True)
    for ax,(ds,phase) in zip(axes,[('D1','development'),('D1','historical'),('D2','development'),('D2','historical')]):
        raw=read(f'SOURCE_DATA/P6/{ds.lower()}_macro_model_comparison.csv');raw=raw[raw.scope.str.contains('DEVELOPMENT' if phase=='development' else ('AUTHOR_TEST' if ds=='D1' else 'HISTORY'))].sort_values('model_id');y=np.arange(len(raw));ax.errorbar(raw.top25_gain_pp,y,xerr=[raw.top25_gain_pp-raw.top25_gain_pp_ci95_low,raw.top25_gain_pp_ci95_high-raw.top25_gain_pp],fmt='o',ms=3,capsize=2,color=BLUE if phase=='development' else ORANGE,lw=.7);ax.set_yticks(y,raw.model_id,fontsize=6.4);ax.invert_yaxis();ax.axvline(0,color=GREY,lw=.7);ax.set_xlabel('Top25 gain (pp)');ax.set_title(ds+' '+phase,fontsize=9);horizontal(ax)
    fig.tight_layout(w_pad=1)
    finish(fig,'Figure_S1_all_P6_configurations',['SOURCE_DATA/P6/d1_macro_model_comparison.csv','SOURCE_DATA/P6/d2_macro_model_comparison.csv'],'All published P6 model configurations, development and historical, including every unfavorable baseline/ablation and random expectation.')
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.7))
    for ax,ds in zip(axes,['D1','D2']):
        q=U[(U.dataset==ds)&(U.phase=='historical')&(U.model_id=='PAIR_ABS')].sort_values('split_group');x=np.arange(len(q));ax.bar(x,q.top25_gain_pp,color=[RED if v<0 else ORANGE for v in q.top25_gain_pp]);ax.axhline(0,color=GREY,lw=.7);ax.set_xticks(x,q.split_group if ds=='D1' else [v[4:6]+'/'+v[6:] for v in q.split_group],rotation=65,fontsize=7);ax.set_ylabel('Top25 PCE gain (pp)');ax.set_title(ds+' previously inspected historical groups',fontsize=9)
    fig.tight_layout();finish(fig,'Figure_S2_all_historical_groups',['ANALYSIS/recomputed_independent_group_metrics.csv'],'All historical group gains for absolute pairwise, with no new evaluation split or claim of blindness.')
