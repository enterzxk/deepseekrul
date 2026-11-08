import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
a=pd.read_csv(HERE/'runs'/'FD002'/'2'/'M0_101.csv')
b=pd.read_csv(HERE/'runs'/'FD002'/'2'/'M0_202.csv')
f=pd.read_csv(HERE/'runs'/'FD002'/'2'/'FULL_101.csv')
m=a.merge(b[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],suffixes=('','_s202'))
m=m.merge(f[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],suffixes=('','_full'))
print('n',len(m))
print('M0 s101            %.3f'%rmse(m))
print('M0 s101+s202 ens   %.3f'%rmse(m.assign(y_pred=(m.y_pred+m.y_pred_s202)/2)))
print('FULL s101          %.3f'%rmse(m.assign(y_pred=m.y_pred_full)))
print('FULL+M0 ens        %.3f'%rmse(m.assign(y_pred=(m.y_pred_full+m.y_pred)/2)))