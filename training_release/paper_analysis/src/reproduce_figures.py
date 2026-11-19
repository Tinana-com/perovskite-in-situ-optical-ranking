"""Regenerate the four main and six SI figures from supplied, archived tables."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import redesign_main_figures as main
import reused_supplement_components as supp
from figure_text_style import finalize_style
B=Path(__file__).resolve().parents[1];OUT=B/'REPRODUCED';PROV=[]
def save(fig,name,sources,note):
    finalize_style(fig,name)
    dest=OUT/('FIGURES_SI' if name.startswith('Figure_S') else 'FIGURES_MAIN');dest.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=300 if name.startswith('Figure_S') else 320,bbox_inches='tight',pad_inches=.12)
    plt.close(fig);PROV.append(dict(figure=name,sources=sources,note=note))
def main_run():
    main.save=save
    for fn in [main.figure1,main.figure2,main.figure3,main.figure4]:fn()
    mapping={'Figure_2_data_protocol':'Figure_S1_data_protocol','Figure_S1_all_P6_configurations':'Figure_S2_all_configurations','Figure_S2_all_historical_groups':'Figure_S4_historical_groups','Figure_4_comparisons_ablations':'Figure_S5_ablations','Figure_5_groups_complexity':'Figure_S6_complexity'}
    supp.finish=lambda fig,n,s,d:save(fig,mapping[n],s,d)
    for fn in [supp.f2,supp.supplemental,supp.f4,supp.f5]:fn()
    # Same four complete comparisons, arranged 2x2 for legibility in portrait SI.
    fig,axs=plt.subplots(2,2,figsize=(8.2,9.0));fig.subplots_adjust(left=.25,right=.985,top=.97,bottom=.07,wspace=1.12,hspace=.27)
    for ax,(ds,phase) in zip(axs.flat,[('D1','development'),('D1','historical'),('D2','development'),('D2','historical')]):
        q=main.read(f'SOURCE_DATA/P6/{ds.lower()}_macro_model_comparison.csv');q=q[q.scope.str.contains('DEVELOPMENT' if phase=='development' else 'AUTHOR_TEST' if ds=='D1' else 'HISTORY')].sort_values('model_id');y=np.arange(len(q));col=supp.BLUE if phase=='development' else supp.ORANGE
        ax.errorbar(q.top25_gain_pp,y,xerr=[q.top25_gain_pp-q.top25_gain_pp_ci95_low,q.top25_gain_pp_ci95_high-q.top25_gain_pp],fmt='o',ms=3,capsize=2,color=col,lw=.7);ax.set_yticks(y,q.model_id,fontsize=7.3);ax.invert_yaxis();ax.axvline(0,color=supp.GREY,lw=.7);ax.set_xlabel('Top25% gain (pp)',fontsize=8);supp.horizontal(ax)
    PROV[:]=[p for p in PROV if p['figure']!='Figure_S2_all_configurations']
    save(fig,'Figure_S2_all_configurations',['SOURCE_DATA/P6/d1_macro_model_comparison.csv','SOURCE_DATA/P6/d2_macro_model_comparison.csv'],'Every original configuration and interval retained; portrait2x2 layout for readable labels. Panel order D1 development/history, D2 development/history.')
    u=main.read('ANALYSIS/recomputed_independent_group_metrics.csv',dtype={'split_group':str});fig,axs=plt.subplots(1,2,figsize=(7.8,3.3))
    for a,ds in zip(axs,['D1','D2']):
        q=u[(u.dataset==ds)&(u.phase=='development')&(u.model_id=='PAIR_ABS')].sort_values('split_group');x=np.arange(len(q));a.bar(x,q.top25_gain_pp,color=[supp.RED if v<0 else supp.BLUE for v in q.top25_gain_pp]);a.axhline(0,color=supp.GREY,lw=.7);a.set_xticks(x,q.split_group if ds=='D1' else [z[4:6]+'/'+z[6:] for z in q.split_group],rotation=90 if ds=='D1' else 65,fontsize=6.5);a.set_ylabel('Top25% gain (pp)')
    fig.tight_layout();save(fig,'Figure_S3_development_groups',['ANALYSIS/recomputed_independent_group_metrics.csv'],'All substrates/dates, no omitted negative groups.')
    (OUT/'figure_provenance.json').write_text(json.dumps(PROV,indent=2),encoding='utf-8')
    print('Regenerated four main and six SI figures; no model fits')
if __name__=='__main__':main_run()
