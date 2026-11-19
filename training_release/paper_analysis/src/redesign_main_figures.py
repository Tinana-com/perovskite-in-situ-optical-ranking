"""Data-led visual replacement of V2 main figures; all measurements are archived."""
from pathlib import Path
import os,json
B=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(B/'REPRODUCED/.mplconfig')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Ellipse,FancyBboxPatch,FancyArrowPatch,Rectangle,Circle
from matplotlib.colors import LogNorm,LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.lines import Line2D
INK='#203A49';TEAL='#0072B2';CORAL='#D55E00';MUTED='#8797A1';PALE='#E8F3F1';GRID='#E7ECEF';NAVY='#38556E'
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':9,'axes.titlesize':11,'xtick.labelsize':8,'ytick.labelsize':8.5,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
PROV=[]
def read(n,**kw):return pd.read_csv(B/n,float_precision='round_trip',**kw)
def save(fig,name,sources,note):
    from figure_text_style import finalize_style
    finalize_style(fig,name)
    for ext in ['png','pdf','svg']:fig.savefig(B/'FIGURES_V2'/f'{name}.{ext}',dpi=320,bbox_inches='tight',pad_inches=.12)
    plt.close(fig);PROV.append(dict(figure=name,sources=sources,note=note))
def title(a,letter,text):
    a.text(0,1.07,f'({letter})',transform=a.transAxes,fontweight='bold',fontsize=12,va='bottom',color='black')
def clean(a,grid=True):
    for n in ['top','right','left']:a.spines[n].set_visible(False)
    a.spines['bottom'].set_color('#BBC7CD');a.tick_params(length=0,pad=5)
    if grid:a.xaxis.grid(True,color=GRID,lw=.65);a.set_axisbelow(True)
def arrow(a,start,end,color=MUTED,lw=1.2):a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,color=color,lw=lw))
def figure1():
    from workflow_figure import draw_workflow
    draw_workflow(B,read,save)

def figure2():
    M=read('ANALYSIS/recomputed_macro_metrics.csv');fig,axs=plt.subplots(1,2,figsize=(8.2,4.5));fig.subplots_adjust(left=.19,right=.985,top=.80,bottom=.15,wspace=.70)
    labels={'PAIR_ABS':'Absolute pairwise','LGBM_PL':'LambdaRank','EXTRATREES_PL':'ExtraTrees','RIDGE_PL':'Reference Ridge','PL_PAIRWISE':'Reference pairwise','ID_PAIRWISE':'Position only','PL_INTENSITY':'Intensity rule','AUTHOR_KNN_AUX':'Author kNN','F2_RIDGE_AUX':'Absolute Ridge'}
    for a,ds,letter in zip(axs,['D1','D2'],'ab'):
        ids=['PAIR_ABS','LGBM_PL','EXTRATREES_PL','PL_PAIRWISE','RIDGE_PL','AUTHOR_KNN_AUX' if ds=='D1' else 'F2_RIDGE_AUX','ID_PAIRWISE','PL_INTENSITY'];y=np.arange(8)
        a.axvline(0,color='#AEBCC5',ls=(0,(3,3)),lw=.8)
        for phase,off,col,fill in [('development',-.12,TEAL,TEAL),('historical',.12,CORAL,'white')]:
            q=M[(M.dataset==ds)&(M.phase==phase)].set_index('model_id').loc[ids]
            a.hlines(y+off,q.original_gain_CI_low,q.original_gain_CI_high,color=col,lw=1.4,alpha=.78)
            a.scatter(q.top25_gain_pp,y+off,s=[30]+[20]*7,color=fill,edgecolor=col,lw=1.2,zorder=4)
        a.set_yticks(y,[labels[m] for m in ids]);a.set_ylim(7.6,-.6);a.set_xlabel('Top25% PCE gain (pp)');clean(a);title(a,letter,ds)
        a.set_xlim((-.2,1.72) if ds=='D1' else (-1.7,2.1))
    fig.legend(handles=[Line2D([0],[0],marker='o',color=TEAL,lw=1.4,ms=5,label='Development'),Line2D([0],[0],marker='o',color=CORAL,mfc='white',lw=1.4,ms=5,label='Historical · previously viewed')],loc='upper center',bbox_to_anchor=(.57,1.00),ncol=2,frameon=False,fontsize=9)
    save(fig,'Figure_2_enrichment',['ANALYSIS/recomputed_macro_metrics.csv','SOURCE_DATA/P6/d1_macro_model_comparison.csv','SOURCE_DATA/P6/d2_macro_model_comparison.csv'],'All same eight original main-figure competitors per dataset; development filled teal and historical open coral with archived descriptive95% intervals. No model or cohort removed.')
def figure3():
    fig=plt.figure(figsize=(8.2,5.3));a=fig.add_axes([.16,.60,.40,.29]);b=fig.add_axes([.74,.60,.235,.29]);pos=read('ANALYSIS/V2_position_summary.csv').set_index(['dataset','phase']).loc[[('D1','development'),('D1','historical'),('D2','development'),('D2','historical')]]
    for y,(_,q) in enumerate(pos.iterrows()):
        col=TEAL if q.delta>0 else CORAL;a.plot([q.low,q.high],[y,y],color=col,lw=1.5);a.scatter(q.delta,y,s=25,color=col,zorder=3)
    a.set_yticks(range(4),['D1 · development','D1 · historical','D2 · development','D2 · historical']);a.set_ylim(3.5,-.5);a.axvline(0,color=MUTED,lw=.7,ls=(0,(3,3)));a.set_xlabel('Δ Top25% gain (pp)');clean(a);title(a,'a','Optical increment over position')
    p=read('ANALYSIS/paired_gain_comparisons.csv');p=p[(p.phase=='development')&(p.method_a=='PL_PAIRWISE')&(p.method_b=='RIDGE_PL')].set_index('dataset').loc[['D1','D2']]
    b.hlines([0,1],p.CI_low,p.CI_high,color=TEAL,lw=1.5);b.scatter(p.delta_gain_pp,[0,1],color=TEAL,s=25,zorder=3);b.set_yticks([0,1],['D1','D2']);b.set_ylim(1.55,-.55);b.axvline(0,color=MUTED,lw=.7,ls=(0,(3,3)));b.set_xlabel('Δ Top25% gain (pp)');clean(b);title(b,'b','Pairwise − Ridge')
    cmap=LinearSegmentedColormap.from_list('gain',[CORAL,'#F8FAFA',TEAL]);norm=TwoSlopeNorm(vmin=-2.1,vcenter=0,vmax=2.1)
    def tiles(rect,ds,phase,letter,label,numbers=False):
        a=fig.add_axes(rect);q=read(f'ANALYSIS/{ds}_{phase}_position_paired.csv',dtype={'split_group':str});vals=q.delta.to_numpy();im=a.imshow([vals],cmap=cmap,norm=norm,aspect='auto',interpolation='nearest');labs=q.split_group if ds=='D1' else [s[4:6]+'/'+s[6:] for s in q.split_group];a.set_xticks(range(len(q)),labs,rotation=90 if ds=='D1' else 50,ha='center' if ds=='D1' else 'right',fontsize=7.5);a.set_yticks([]);a.tick_params(length=0,pad=4)
        for s in a.spines.values():s.set_visible(False)
        a.set_xticks(np.arange(-.5,len(q),1),minor=True);a.grid(which='minor',axis='x',color='white',lw=1.2);a.tick_params(which='minor',length=0);a.annotate(f'({letter})',(0,1.42),xycoords='axes fraction',xytext=(0,0),textcoords='offset points',fontweight='bold',fontsize=12,color='black')
        if numbers:
            for i,v in enumerate(vals):a.text(i,0,f'{v:+.3f}',ha='center',va='center',fontsize=8,color=INK)
        return im
    im=tiles([.16,.36,.815,.065],'D1','development','c','D1 · every development substrate')
    tiles([.16,.125,.505,.075],'D2','development','d','D2 · development dates')
    tiles([.77,.125,.205,.075],'D2','historical','e','D2 · historical',True)
    cax=fig.add_axes([.46,.018,.28,.012]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_ticks([-2,0,2]);cb.ax.tick_params(labelsize=7,length=0,pad=2);cb.outline.set_visible(False);fig.text(.16,.012,'Matched optical increment (pp)',fontsize=8,color=MUTED)
    save(fig,'Figure_3_matched_controls',['ANALYSIS/V2_position_summary.csv','ANALYSIS/paired_gain_comparisons.csv','ANALYSIS/D1_development_position_paired.csv','ANALYSIS/D2_development_position_paired.csv','ANALYSIS/D2_historical_position_paired.csv'],'Identical position and estimator contrasts; all group differences retained in diverging tiles with a shared scale. Same group-bootstrap intervals; all three historical D2 dates explicitly negative.')
def figure4():
    traces=read('ANALYSIS/real_LP725_examples.csv');meta=read('ANALYSIS/real_LP725_example_metadata.csv');regions=read('ANALYSIS/real_example_all_regions.csv');fig,axs=plt.subplots(2,2,figsize=(8.2,5.5));fig.subplots_adjust(left=.09,right=.985,bottom=.10,top=.84,hspace=.47,wspace=.27)
    for j,sid in enumerate(['ACD','AEM']):
        a=axs[0,j];q=meta[meta.substrate==sid]
        for _,r in q.iterrows():
            t=traces[traces.sample_id==r.sample_id];col=TEAL if r.extremum=='highest_PCE' else CORAL;a.plot(t.frame_index,t.LP725_intensity,color=col,lw=1.15)
        a.set(xlim=(0,718),ylim=(0,4000),xlabel='Recorded frame',ylabel='LP725 intensity (a.u.)');a.set_yticks([0,2000,4000],['0','2,000','4,000']);clean(a,False);title(a,chr(97+j),sid)
        a=axs[1,j];g=regions[regions.unit_id==sid].sort_values(['PAIR_ABS','sample_id'],ascending=[False,True],kind='stable');k=int(np.ceil(len(g)/4));x=np.arange(1,len(g)+1);a.axhline(g.target.mean(),color=MUTED,lw=.8,ls=(0,(3,3)));a.axhline(g.target.iloc[:k].mean(),color=TEAL,lw=1.15);a.scatter(x,g.target,c=[TEAL]*k+['#A6B3BA']*(len(g)-k),s=20,edgecolor='white',lw=.4,zorder=3);a.set(xlim=(.5,len(g)+.5),ylim=(0,15),xlabel='Predicted rank',ylabel='Measured PCE (%)');a.set_yticks([0,5,10,15]);clean(a,False);title(a,chr(99+j),f'{sid} · top {k} of {len(g)}')
    fig.legend(handles=[Line2D([0],[0],color=TEAL,lw=1.15,label='Highest-PCE region'),Line2D([0],[0],color=CORAL,lw=1.15,label='Lowest-PCE region')],loc='upper center',bbox_to_anchor=(.55,1),ncol=2,frameon=False,fontsize=9)
    save(fig,'Figure_4_real_trajectories',['ANALYSIS/real_LP725_examples.csv','ANALYSIS/real_LP725_example_metadata.csv','ANALYSIS/real_example_all_regions.csv'],'Same original four719-frame LP725 traces and all60 candidate scores/PCEs. Common intensity and PCE axes, same extrema and median-positive/sole-negative examples; all failures retained.')
def draw_main():
    for f in [figure1,figure2,figure3,figure4]:f()
    p=B/'FIGURES_V2/figure_provenance.json';old=json.loads(p.read_text()) if p.exists() else [];allrows=PROV+[r for r in old if not r['figure'].startswith(('Figure_1_','Figure_2_','Figure_3_','Figure_4_'))];p.write_text(json.dumps(allrows,indent=2),encoding='utf-8');print('Replaced all four main figures; original measurements retained.')
if __name__=='__main__':draw_main()
