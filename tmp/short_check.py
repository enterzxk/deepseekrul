import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
rows=[]
for k in [0,1,2]:
    a=HERE/'runs'/'FD002'/str(k)/'FULL_101.csv'; b=HERE/'runs3_w100_w100'/'FD002'/str(k)/'FULL_101.csv'
    if not b.is_file(): continue
    w50=pd.read_csv(a); w100=pd.read_csv(b)
    hyb=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
    sel=(hyb.end_cycle>=100)&hyb.y_pred_t.notna()
    hyb.loc[sel.values,'y_pred']=hyb.loc[sel,'y_pred_t'].values
    for grp,sub in [('short (<=Q1 life)',hyb[hyb.short]),('long (>Q1)',hyb[~hyb.short])]:
        rows.append((k,grp,rmse(sub.assign(y_pred=sub.y_pred)),rmse(sub.assign(y_pred=sub.y_pred))))
    for grp,mask in [('short',hyb.short),('long',~hyb.short)]:
        base=hyb[mask]; ref=w50[mask]
        print('fold %d %-6s n=%5d  w50 %.3f -> sched %.3f (%+.1f%%)'%(k,grp,len(base),rmse(ref),rmse(base),100*(rmse(base)/rmse(ref)-1)))