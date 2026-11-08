"""Per-fold RMSE table for the paper."""
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(p):
    d=pd.read_csv(p); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
rows=[]
for subset in ['FD002','FD004']:
    for arm in ['M0','OPS','FULL','NORM']:
        vals=[]
        for k in range(5):
            p=HERE/'runs'/subset/str(k)/f'{arm}_101.csv'
            vals.append(rmse(p) if p.is_file() else np.nan)
        rows.append(dict(subset=subset, arm=arm, **{f'f{k}':round(v,3) if np.isfinite(v) else None for k,v in enumerate(vals)}))
df=pd.DataFrame(rows)
df.to_csv(HERE/'paper_tables'/'per_fold.csv', index=False)
print(df.to_string(index=False))
