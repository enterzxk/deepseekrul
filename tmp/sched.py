import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
def load(p): return pd.read_csv(p) if Path(p).is_file() else None
w50m=load(HERE/'runs'/'FD002'/'2'/'M0_101.csv')
w100m=load(HERE/'runs3_w100_w100'/'FD002'/'2'/'M0_101.csv')
w150m=load(HERE/'runs3_w150_w150'/'FD002'/'2'/'M0_101.csv')
w50f=load(HERE/'runs'/'FD002'/'2'/'FULL_101.csv')
w100f=load(HERE/'runs3_w100_w100'/'FD002'/'2'/'FULL_101.csv')
w150f=load(HERE/'runs3_w150_w150'/'FD002'/'2'/'FULL_101.csv')
def schedule(base, tiers):
    out=base.copy()
    for cut, d in tiers:
        if d is None: continue
        mm=base.merge(d[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
        sel=(mm.end_cycle>=cut) & mm.y_pred_t.notna()
        out.loc[sel.values,'y_pred']=mm.loc[sel,'y_pred_t'].values
    return out
print('w50 M0 %.3f  FULL %.3f  (all %d windows)'%(rmse(w50m),rmse(w50f),len(w50m)))
for tag, tiers in [('2-tier 100', [(100,w100m)]), ('3-tier 100/150', [(100,w100m),(150,w150m)])]:
    s=schedule(w50m, tiers)
    print('M0 %-16s %.3f  n=%d  vs w50 %+.1f%%   vs FULL %+.1f%%'%(tag,rmse(s),len(s),100*(rmse(s)/rmse(w50m)-1),100*(rmse(s)/rmse(w50f)-1)))
for tag, tiers in [('2-tier 100', [(100,w100f)]), ('3-tier 100/150', [(100,w100f),(150,w150f)])]:
    s=schedule(w50f, tiers)
    print('FULL %-14s %.3f  n=%d  vs FULL %+.1f%%'%(tag,rmse(s),len(s),100*(rmse(s)/rmse(w50f)-1)))