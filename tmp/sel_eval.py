import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def met(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return np.sqrt(g.e2.mean()),g.ae.mean(),g.s.mean(),g.b.mean()
for tag,label in [('SEL_best_m0','select=best, no mixup'),('SEL_best_m0.2','select=best, mixup')]:
    ds=[]
    for k in range(5):
        p=H/f'runs_v3_{tag}'/'FD002'/str(k)/'M0_101.csv'
        if p.is_file(): ds.append(pd.read_csv(p))
    if not ds: print(label,'none'); continue
    r=met(pd.concat(ds,ignore_index=True))
    print('%-26s folds=%d RMSE %.3f MAE %.3f NASA %.2f bias %+.3f'%(label,len(ds),r[0],r[1],r[2],r[3]))
print()
print('--- reference: frozen protocol (select=final) ---')
for tag,label in [('DM_w50','final, no mixup'),('DM_w50_m0.2','final, mixup')]:
    ds=[]
    for k in range(5):
        p=H/f'runs_v3_{tag}'/'FD002'/str(k)/'M0_101.csv'
        if p.is_file(): ds.append(pd.read_csv(p))
    if not ds: continue
    r=met(pd.concat(ds,ignore_index=True))
    print('%-26s folds=%d RMSE %.3f'%(label,len(ds),r[0]))