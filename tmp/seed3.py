"""Three-seed summary for FD002 using MATCHED folds (both arms present)."""
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(p):
    d=pd.read_csv(p); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
rows=[]
for s in [101,202,303]:
    folds=[]
    for k in range(5):
        m=HERE/'runs/FD002'/str(k)/f'M0_{s}.csv'; f=HERE/'runs/FD002'/str(k)/f'FULL_{s}.csv'
        if m.is_file() and f.is_file():
            folds.append((k,rmse(m),rmse(f)))
    if folds:
        m0=np.mean([x[1] for x in folds]); fu=np.mean([x[2] for x in folds])
        rows.append(dict(seed=s,matched_folds=len(folds),fold_ids=''.join(str(x[0]) for x in folds),
                         M0=m0,FULL=fu,delta=fu-m0,rel=100*(fu/m0-1)))
d=pd.DataFrame(rows)
print(d.round(3).to_string(index=False))
print('\nacross %d seeds (matched folds per seed):'%len(d))
print('  M0 mean %.3f sd %.3f'%(d.M0.mean(),d.M0.std(ddof=1)))
print('  FULL mean %.3f sd %.3f'%(d.FULL.mean(),d.FULL.std(ddof=1)))
print('  relative: mean %+.2f%% sd %.2f%%'%(d.rel.mean(),d.rel.std(ddof=1)))
