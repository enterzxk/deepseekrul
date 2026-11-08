import pandas as pd, numpy as np
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(f):
    d=pd.read_csv(f); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
print('BASE 42k M0 f2: %.3f'%rmse(H/'runs/FD002/2/M0_101.csv'))
for tag in ['A','B','AB','BASE']:
    f=H/f'runs_v3_{tag}/FD002/2/M0_101.csv'
    if f.is_file(): print('%s M0 f2: %.3f'%(tag,rmse(f)))