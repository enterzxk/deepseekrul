"""Matched-fold multi-seed comparison (folds where both seeds are complete)."""
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(p):
    d=pd.read_csv(p); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
rows=[]
for k in range(5):
    r={'fold':k}
    for s in [101,202,303]:
        for a in ['M0','FULL']:
            p=HERE/'runs/FD002'/str(k)/f'{a}_{s}.csv'
            r[f'{a}_{s}']=rmse(p) if p.is_file() else np.nan
    rows.append(r)
d=pd.DataFrame(rows)
print(d.round(3).to_string(index=False))
for s in [101,202,303]:
    sub=d.dropna(subset=[f'M0_{s}',f'FULL_{s}'])
    if len(sub)==0: continue
    m0=sub[f'M0_{s}'].mean(); fu=sub[f'FULL_{s}'].mean()
    print('seed %d: %d folds  M0 %.3f  FULL %.3f  FULL-M0 %+.3f (%+.2f%%)'%(s,len(sub),m0,fu,fu-m0,100*(fu/m0-1)))
# matched folds 0-3 across seeds 101 and 202
sub=d.dropna(subset=['M0_101','FULL_101','M0_202','FULL_202'])
if len(sub):
    print('matched folds', list(sub.fold))
    for s in [101,202]:
        m0=sub[f'M0_{s}'].mean(); fu=sub[f'FULL_{s}'].mean()
        print('  seed %d matched: M0 %.3f FULL %.3f  %+.2f%%'%(s,m0,fu,100*(fu/m0-1)))
