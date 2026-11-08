import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
for seed in [101,202]:
    print('=== seed',seed,'FD004 ===')
    for k in range(5):
        a=HERE/'runs'/'FD004'/str(k)/f'FULL_{seed}.csv'
        b=HERE/'runs3_w100_w100'/'FD004'/str(k)/f'FULL_{seed}.csv'
        if not (a.is_file() and b.is_file()): continue
        w50=pd.read_csv(a); w100=pd.read_csv(b)
        d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
        sel=(d.end_cycle>=100)&d.y_pred_t.notna()
        d.loc[sel.values,'y_pred']=d.loc[sel,'y_pred_t'].values
        e=d.y_pred-d.y_true; e50=w50.y_pred-w50.y_true
        print(' fold %d: sched NASA %.2f  ref NASA %.2f  max|err| sched %.1f ref %.1f'%(
            k, nasa(e).mean(), nasa(e50).mean(), e.abs().max(), e50.abs().max()))
        top=d.assign(s=nasa(e)).groupby('unit').s.mean().sort_values(ascending=False).head(2)
        print('        worst units:', [(int(i), round(v,1)) for i,v in top.items()])