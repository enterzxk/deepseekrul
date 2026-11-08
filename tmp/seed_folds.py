from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(p):
    d=pd.read_csv(p); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
print('FD002 per-fold M0 across seeds (run-to-run variability):')
for k in range(5):
    vals=[]
    for s in [101,202,303]:
        p=HERE/'runs/FD002'/str(k)/f'M0_{s}.csv'
        vals.append('%.2f'%rmse(p) if p.is_file() else '--')
    print('  fold %d: %s'%(k,'  '.join(vals)))
print('FD002 per-fold FULL across seeds:')
for k in range(5):
    vals=[]
    for s in [101,202,303]:
        p=HERE/'runs/FD002'/str(k)/f'FULL_{s}.csv'
        vals.append('%.2f'%rmse(p) if p.is_file() else '--')
    print('  fold %d: %s'%(k,'  '.join(vals)))
