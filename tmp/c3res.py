import pandas as pd, numpy as np
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(f):
    d=pd.read_csv(f); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
rows=[('42k baseline d64/1',H/'runs/FD002/2/M0_101.csv'),
      ('C3 d64/1 + modules only (116k)',H/'runs_sweep_c3/FD002/2/M0_101.csv'),
      ('C1 d96/1 + modules (258k)',H/'runs_sweep_c1/FD002/2/M0_101.csv'),
      ('deep d128/2 + modules (719k)',H/'runs_deep_w50_deep/FD002/2/M0_101.csv')]
for n,f in rows:
    print('%-36s %s'%(n, ('%.3f'%rmse(f)) if f.is_file() else 'missing'))