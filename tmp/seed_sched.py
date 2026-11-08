"""Schedule metrics for an arbitrary seed (per-fold + covered estimate)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
seed=int(sys.argv[1]) if len(sys.argv)>1 else 202
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
for subset in ['FD002','FD004']:
    for arm in ['M0','FULL']:
        per=[]
        for k in range(5):
            a=HERE/'runs'/subset/str(k)/f'{arm}_{seed}.csv'
            b=HERE/'runs3_w100_w100'/subset/str(k)/f'{arm}_{seed}.csv'
            if not a.is_file() or not b.is_file(): continue
            w50=pd.read_csv(a); w100=pd.read_csv(b)
            d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
            sel=(d.end_cycle>=100)&d.y_pred_t.notna()
            d.loc[sel.values,'y_pred']=d.loc[sel,'y_pred_t'].values
            per.append((k, rmse(w50), rmse(d)))
        if not per: continue
        rel=100*(np.mean([p[2] for p in per])/np.mean([p[1] for p in per])-1)
        print('seed %d %s %-4s folds %s  w50 %.3f -> sched %.3f  %+.2f%%'%(
            seed,subset,arm,''.join(str(p[0]) for p in per),
            np.mean([p[1] for p in per]),np.mean([p[2] for p in per]),rel))