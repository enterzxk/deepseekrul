"""FD004 NASA anomaly: which engines dominate the asymmetric score?"""
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def load(arm):
    b=[]
    for k in range(5):
        p=HERE/'runs/FD004'/str(k)/f'{arm}_101.csv'
        if not p.is_file(): return None
        b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b,ignore_index=True)
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
for arm in ['M0','OPS','FULL']:
    d=load(arm)
    if d is None: continue
    d['err']=d.y_pred-d.y_true; d['s']=nasa(d.err)
    g=d.groupby('unit').agg(nasa=('s','mean'),bias=('err','mean'),rmse=('err',lambda x: float(np.sqrt((x**2).mean()))))
    tot=g.nasa.sum()
    top=g.sort_values('nasa',ascending=False).head(5)
    print('== %s  total per-engine NASA sum %.1f  mean %.2f'%(arm,tot,g.nasa.mean()))
    print('   top5 units:', list(top.index), 'nasa', np.round(top.nasa.values,1), 'bias', np.round(top.bias.values,1))
    print('   frac of sum from top5: %.2f'%(top.nasa.sum()/tot))
