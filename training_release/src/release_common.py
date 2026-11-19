"""Portable IO and loading of byte-preserved frozen scientific implementations."""
from pathlib import Path
import os,sys,json,hashlib
B=Path(__file__).resolve().parents[1]
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR',str(B/'outputs/.mplconfig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load_core():
    sys.path.insert(0,str(B/'frozen_sources/P6/src'))
    import p6_core
    return p6_core
def fresh_output(path):
    p=Path(path).resolve()
    protected=[B/'frozen_sources',B/'reference_results',B/'paper_analysis',B/'data_cache',B/'configs',B/'src',B/'tests']
    if p==B or any(p.is_relative_to(q.resolve()) for q in protected):raise ValueError('Use a fresh output directory outside source/reference inputs')
    if p.exists() and any(p.iterdir()):raise FileExistsError('Existing attempt preserved; choose a new output path')
    p.mkdir(parents=True,exist_ok=True);return p
