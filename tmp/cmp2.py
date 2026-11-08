import pandas as pd, numpy as np
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(f):
    d=pd.read_csv(f); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
pairs=[('M0',H/'runs/FD002/2/M0_101.csv',H/'runs_deep_w50_deep/FD002/2/M0_101.csv'),
       ('FULL',H/'runs/FD002/2/FULL_101.csv',H/'runs_deep_w50_deep/FD002/2/FULL_101.csv')]
for arm,a,b in pairs:
    print('%-5s 42k %.3f -> deep719k %s'%(arm,rmse(a), ('%.3f'%rmse(b)) if b.is_file() else 'missing'))