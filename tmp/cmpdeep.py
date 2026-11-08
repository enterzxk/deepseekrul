import pandas as pd, numpy as np
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
b=H/'runs_deep_w50_deep/FD002/2/M0_101.csv'
if b.is_file():
    d=pd.read_csv(b); print('deep M0 fold2: %.3f  (42k baseline 13.732)'%rmse(d))
o=H/'runs/FD002/2/M0_101.csv'
print('42k M0 fold2: %.3f'%rmse(pd.read_csv(o)))