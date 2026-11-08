import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
def load(p): return pd.read_csv(p) if Path(p).is_file() else None
base=pd.read_csv(HERE/'runs'/'FD002'/'2'/'FULL_101.csv')
cols={}
for s in [101,202,303]:
    p=HERE/'runs'/'FD002'/'2'/f'FULL_{s}.csv'
    if p.is_file():
        d=pd.read_csv(p); m=base[['unit','end_cycle']].merge(d[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left')
        cols[s]=m.y_pred.to_numpy()
        print('FULL seed %d: %.3f'%(s, rmse(base.assign(y_pred=m.y_pred))))
if len(cols)>=2:
    ens=np.mean(np.stack(list(cols.values())),axis=0)
    print('FULL %d-seed ensemble: %.3f'%(len(cols), rmse(base.assign(y_pred=ens))))
print('schedule FULL 3-tier: 11.802' if False else 'schedule FULL 3-tier (from sched.py): 11.802')