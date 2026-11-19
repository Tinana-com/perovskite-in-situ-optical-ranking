"""Requested journal figure style: panel labels only, every text glyph black."""
from pathlib import Path
import json
import re
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.colors import to_rgba,LinearSegmentedColormap
B=Path(__file__).resolve().parents[1]
def finalize_style(fig,name):
    plt.rcParams.update({'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'})
    axes=[a for a in fig.axes if not getattr(a,'_colorbar',None)]
    for a in axes:
        for spine in a.spines.values():spine.set_color('black');spine.set_linewidth(.6)
        a.tick_params(axis='both',width=.6,direction='out')
    if name.startswith('Figure_S'):
        for i,a in enumerate(axes):
            a.set_title('');a.set_title('',loc='right');a.set_title(f'({chr(97+i)})',loc='left',fontweight='bold',color='black')
    # Numbered fold cells use light colors so all-black labels remain legible.
    if name=='Figure_S1_data_protocol':
        light=LinearSegmentedColormap.from_list('fold_light',plt.colormaps['Blues']([.06,.48]))
        for a in axes:
            for im in a.images:im.set_cmap(light)
            if a.get_xlabel().startswith('26 substrates'):a.set_xlabel('Substrate ID')
            if a.get_xlabel().startswith('14 dates'):a.set_xlabel('Date')
    fig.canvas.draw()
    labels=list(fig.findobj(match=Text))
    for t in labels:
        t.set_color('black');t.set_fontweight('normal')
        if re.fullmatch(r'\([a-z]\)',t.get_text()):t.set_fontsize(10);t.set_fontweight('bold')
    assert all(to_rgba(t.get_color())==to_rgba('black') for t in labels)
    p=B/'REPRODUCED/RUN_RECEIPTS/figure_text_style.json';p.parent.mkdir(parents=True,exist_ok=True);r=json.loads(p.read_text()) if p.exists() else {}
    r[name]={'all_text_black':True,'text_objects':len(labels),'panel_titles':'letters only','renderer':'Python Matplotlib','style':'thin black axes; regular-weight text; white background','fold_palette_lightened':name=='Figure_S1_data_protocol'}
    p.write_text(json.dumps(r,indent=2),encoding='utf-8')
