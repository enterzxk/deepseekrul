import pandas as pd, numpy as np, glob
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def rmse(f):
    d=pd.read_csv(f); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean())), len(g)
notes={'A':'trajectory monotonicity loss (lam1=lam2=0.5)','B':'metric-matched pinball loss (tau=0.9, lam3=0.1)',
       'C':'lifetime reparameterisation (predict L, RUL=min(125,L-t))','D':'engine-level mixup (alpha=0.2, all channels)',
       'DX':'mixup on inputs only (prior not mixed)','graph':'cross-sensor graph attention (~200 params)'}
rows=[]
for f in sorted(glob.glob(str(H/'runs_v3_*'/'FD00*'/'*'/'*_101.csv')))+sorted(glob.glob(str(H/'runs3_w50_graph'/'FD00*'/'*'/'*_101.csv'))):
    p=Path(f); arm=p.name.split('_')[0]
    if p.parents[2].name.startswith('runs3_w50_graph'): tag='E'
    else: tag=p.parents[2].name.replace('runs_v3_','')
    subset=p.parents[1].name; fold=p.parents[0].name
    b=H/'runs'/subset/fold/f'{arm}_101.csv'
    if not b.is_file(): continue
    rv,n=rmse(p); rb,_=rmse(b)
    rows.append(dict(subset=subset,variant=f'{tag}_{arm}',folds=1,note=notes.get(tag,tag)+f' (fold {fold})',
        rmse_variant=round(rv,4),rmse_M0=round(rb,4),delta=round(rv-rb,4),ci_lo='',ci_hi='',engines=n))
if rows:
    out=pd.DataFrame(rows); out.to_csv(H/'ABLATIONS_V3.csv',index=False)
    pd.set_option('display.width',220); print(out.to_string(index=False))
else: print('none')