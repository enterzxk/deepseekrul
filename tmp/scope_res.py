import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def met(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return np.sqrt(g.e2.mean()), g.ae.mean(), g.s.mean(), g.b.mean()
for subset in ['FD001','FD003']:
    dm=[];fu=[];pe=[]
    for k in range(5):
        b=H/'runs_v3_DM_w50'/subset/str(k)/'M0_101.csv'
        a=H/'runs'/subset/str(k)/'FULL_101.csv'
        p=H/'runs'/subset/str(k)/'M0_101.csv'
        if b.is_file(): dm.append(pd.read_csv(b))
        if a.is_file(): fu.append(pd.read_csv(a))
        if p.is_file(): pe.append(pd.read_csv(p))
    if not dm: print(subset,'none'); continue
    r=met(pd.concat(dm,ignore_index=True))
    s='%s DM(w50) %d folds: RMSE %.3f'%(subset,len(dm),r[0])
    if fu: s+='  vs FULL %.3f (%+.2f%%)'%(met(pd.concat(fu,ignore_index=True))[0], 100*(r[0]/met(pd.concat(fu,ignore_index=True))[0]-1))
    if pe: s+='  vs PE %.3f (%+.2f%%)'%(met(pd.concat(pe,ignore_index=True))[0], 100*(r[0]/met(pd.concat(pe,ignore_index=True))[0]-1))
    print(s)