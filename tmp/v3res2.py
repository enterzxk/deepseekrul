import pandas as pd, numpy as np
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def met(f):
    d=pd.read_csv(f); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return np.sqrt(g.e2.mean()), g.ae.mean(), g.s.mean(), g.b.mean()
rows=[('BASE',H/'runs/FD002/2/M0_101.csv')]
for tag in ['A','B','AB','BASE']:
    f=H/f'runs_v3_{tag}/FD002/2/M0_101.csv'
    if f.is_file(): rows.append((tag,f))
for n,f in rows:
    r=met(f); print('%-5s RMSE %.3f  MAE %.3f  NASA %.2f  bias %+.3f'%(n,*r))