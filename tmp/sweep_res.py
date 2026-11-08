import pandas as pd, numpy as np
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(f):
    d=pd.read_csv(f); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
rows=[('42k baseline',H/'runs/FD002/2/M0_101.csv'),
      ('deep 719k d128L2',H/'runs_deep_w50_deep/FD002/2/M0_101.csv'),
      ('C1 258k d96L1 ms+attn reg',H/'runs_sweep_c1/FD002/2/M0_101.csv'),
      ('C2 719k d128L2 ms+attn reg',H/'runs_sweep_c2/FD002/2/M0_101.csv')]
for name,f in rows:
    print('%-28s %s'%(name, ('%.3f'%rmse(f)) if f.is_file() else 'missing'))