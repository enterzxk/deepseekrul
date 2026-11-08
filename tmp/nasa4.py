import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def load(subset, fold, arm, seed):
    a=HERE/'runs'/subset/str(fold)/f'{arm}_{seed}.csv'
    b=HERE/'runs3_w100_w100'/subset/str(fold)/f'{arm}_{seed}.csv'
    if not a.is_file() or not b.is_file(): return None,None
    w50=pd.read_csv(a); w100=pd.read_csv(b)
    d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
    sel=(d.end_cycle>=100)&d.y_pred_t.notna()
    d.loc[sel.values,'y_pred']=d.loc[sel,'y_pred_t'].values
    return d, w50
for seed in [101,202]:
    S=[];R=[]
    for k in range(5):
        s,r=load('FD004',k,'FULL',seed)
        if s is None: continue
        S.append(s.assign(fold=k)); R.append(r.assign(fold=k))
    S=pd.concat(S,ignore_index=True); R=pd.concat(R,ignore_index=True)
    for name,d in [('sched',S),('ref',R)]:
        e=d.y_pred-d.y_true
        g=pd.DataFrame({'u':d.unit,'s':nasa(e),'b':e}).groupby('u').mean()
        top=g.sort_values('s',ascending=False).head(3)
        print('seed %d FD004 %-5s NASA mean %.2f  top3 %s  frac %.2f'%(
            seed,name,g.s.mean(),[int(i) for i in top.index],top.s.sum()/g.s.sum()))