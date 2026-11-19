"""Original Matplotlib workflow artwork; measured traces only in panel c."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon,Circle,FancyArrowPatch
BLUE='#0072B2';ORANGE='#D55E00';GREY='#A6B3BA';PALE='#EAF3F8'
def draw_workflow(B,read,save):
    plt.rcParams.update({'font.family':'Arial','font.size':9,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(9.0,6.15));a=fig.add_axes([.03,.55,.45,.41]);b=fig.add_axes([.53,.55,.44,.41]);c=fig.add_axes([.03,.06,.45,.42]);d=fig.add_axes([.53,.06,.44,.42])
    def label(ax,ch):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(0,1.02,'('+ch+')',fontsize=11,fontweight='bold',va='bottom')
    for ax,ch in zip([a,b,c,d],'abcd'):label(ax,ch)
    def txt(ax,x,y,s,fs=9,**kw):ax.text(x,y,s,fontsize=fs,color='black',ha=kw.pop('ha','center'),va='center',**kw)
    def rect(ax,x,y,w,h,fill='white',lw=.8):ax.add_patch(Rectangle((x,y),w,h,fc=fill,ec='black',lw=lw))
    def arrow(ax,p,q):ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=9,color='black',lw=.8))
    def camera(ax,x,y):
        rect(ax,x,y,.17,.13);ax.add_patch(Circle((x+.085,y+.065),.032,fc=PALE,ec='black',lw=.8));rect(ax,x+.025,y+.13,.055,.025)
    # D1: illustrative regions, never a claimed measured spatial map.
    txt(a,.10,.92,'D1',fs=11,ha='left');txt(a,.61,.92,'LP725')
    camera(a,.13,.65);arrow(a,(.215,.64),(.215,.47))
    poly=np.array([[.05,.16],[.40,.16],[.46,.43],[.11,.43]])
    a.add_patch(Polygon(poly,fc=PALE,ec='black',lw=.9))
    for row in range(4):
        for col in range(4):
            x=.095+col*.072+row*.012;y=.19+row*.052
            a.add_patch(Polygon([[x,y],[x+.047,y],[x+.054,y+.032],[x+.007,y+.032]],fc='white',ec=BLUE,lw=.65))
    arrow(a,(.47,.30),(.59,.30));rect(a,.62,.19,.29,.22)
    for i in range(3):a.plot([.66,.69,.73],[.35-i*.06,.32-i*.06,.36-i*.06],color=BLUE,lw=1)
    txt(a,.81,.30,'PCE',fs=10)
    txt(a,.25,.08,'Device-associated regions',fs=8.7);txt(a,.77,.08,'Region labels',fs=8.7)
    txt(a,.66,.70,'780 regions\n26 development substrates',fs=9,linespacing=1.6)
    # D2: categorical ROI identities; no fabricated physical layout or filters.
    txt(b,.10,.92,'D2',fs=11,ha='left');camera(b,.11,.66)
    for i,(text,col) in enumerate([('LP725',BLUE),('LP780',BLUE),('SP775',BLUE),('ND',ORANGE)]):
        x=.37+i*.145;rect(b,x,.65,.13,.15,PALE if col==BLUE else '#FCF0E8');txt(b,x+.065,.725,text,fs=8)
    arrow(b,(.28,.725),(.36,.725));arrow(b,(.565,.62),(.565,.51))
    for i in range(3):rect(b,.09+i*.026,.24+i*.025,.18,.18)
    txt(b,.20,.16,'ROI / patch ID',fs=8.5);arrow(b,(.36,.35),(.44,.35))
    rect(b,.46,.26,.20,.18);txt(b,.56,.35,'Experiment',fs=8);arrow(b,(.68,.35),(.75,.35));rect(b,.77,.26,.16,.18);txt(b,.85,.35,'Date',fs=8.5)
    txt(b,.53,.04,'4,707 regions · 158 experiments · 14 dates',fs=8.8)
    # Analysis pipeline: real archived ACD transients in the left inset.
    t=read('ANALYSIS/figure1_ACD_LP725.csv');tr=fig.add_axes([.052,.235,.17,.14])
    for sid,g in t.groupby('sample_id'):
        col=ORANGE if sid=='D1::ACD::74' else BLUE if sid=='D1::ACD::43' else '#C3CDD2'
        tr.plot(g.frame_index,g.LP725_intensity,color=col,lw=1 if col in [BLUE,ORANGE] else .45,alpha=1 if col in [BLUE,ORANGE] else .6,zorder=3 if col in [BLUE,ORANGE] else 1)
    tr.set(xlim=(0,718),ylim=(0,4000));tr.set_xticks([0,350,700]);tr.set_yticks([0,4000],['0','4k']);tr.tick_params(labelsize=7,length=2,width=.6);tr.spines[['top','right']].set_visible(False)
    tr.set_xlabel('Recorded frame',fontsize=8,labelpad=2);tr.set_ylabel('PL (a.u.)',fontsize=8,labelpad=2)
    txt(c,.22,.91,'Optical trajectories',fs=9);txt(c,.61,.91,'Features',fs=9);txt(c,.22,.19,'Measured ACD example',fs=8)
    arrow(c,(.43,.54),(.49,.54))
    for row in range(3):
        for col in range(4):rect(c,.52+col*.035,.44+row*.071,.025,.045,PALE)
    txt(c,.595,.29,'12 summaries',fs=8.5);arrow(c,(.69,.54),(.78,.665));arrow(c,(.69,.54),(.78,.415))
    rect(c,.79,.59,.20,.15);txt(c,.89,.665,'Ranking',fs=8.5);rect(c,.79,.34,.20,.15);txt(c,.89,.415,'Regression',fs=8)
    txt(c,.50,.04,'Separate models for D1 and D2',fs=8.8)
    # Evaluation: symbolic folds and upper-quarter choice; no simulated outcome.
    txt(d,.50,.92,'Group-isolated cross-validation',fs=9.5)
    for i in range(5):rect(d,.08+i*.105,.69,.082,.09,ORANGE if i==4 else PALE)
    txt(d,.25,.60,'Training groups',fs=8);txt(d,.57,.60,'Held-out',fs=8)
    arrow(d,(.34,.53),(.34,.45))
    for i in range(8):rect(d,.08+i*.068,.30,.05,.10,BLUE if i<2 else 'white')
    txt(d,.33,.20,'k = ceil(n/4)',fs=8.7);arrow(d,(.64,.35),(.71,.35))
    txt(d,.86,.49,'Spearman ρ',fs=8.5);txt(d,.86,.35,'Gain (pp)',fs=8.5);txt(d,.86,.21,'Overlap',fs=8.5)
    txt(d,.50,.04,'Position-only  /  optical-only  /  combined',fs=8.6)
    # Black typography throughout; colored marks encode modalities and subsets.
    from matplotlib.text import Text
    for t in fig.findobj(match=Text):t.set_color('black')
    save(fig,'Figure_1_framework',['ANALYSIS/figure1_ACD_LP725.csv'],'Original Python vector schematics in a,b,d; original graphics and measured ACD LP725 curves in c. Device/ROI geometry and fold icons are schematic, not measured maps. No external image assets or image-generation service.')
