"""Does a monotone (isotonic) trajectory constraint help? Test on saved predictions."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def load(subset, arm, seed=101):
    b=[]
    for k in range(5):
        p=HERE/'runs'/subset/str(k)/f'{arm}_{seed}.csv'
        if not p.is_file(): return None
        b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b, ignore_index=True)
def pooled(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
def smooth(d):
    out=[]
    for (f,u),g in d.groupby(['fold','unit']):
        g=g.sort_values('end_cycle')
        if len(g)<3:
            out.append(g); continue
        ir=IsotonicRegression(increasing=False, out_of_bounds='clip')
        g=g.copy()
        g['y_pred']=ir.fit_transform(g.end_cycle.to_numpy(float), g.y_pred.to_numpy(float))
        out.append(g)
    return pd.concat(out, ignore_index=True)
for subset in ['FD002','FD004']:
    for arm in ['M0','FULL']:
        d=load(subset,arm)
        if d is None: continue
        raw=pooled(d); sm=pooled(smooth(d))
        print('%s %-4s raw %.3f  isotonic %.3f  (%+.2f%%)'%(subset,arm,raw,sm,100*(sm/raw-1)))