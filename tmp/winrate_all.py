"""Per-engine win rates on all four subsets."""
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def load(subset,arm):
    b=[]
    for k in range(5):
        p=HERE/'runs'/subset/str(k)/f'{arm}_101.csv'
        if not p.is_file(): return None
        b.append(pd.read_csv(p))
    return pd.concat(b,ignore_index=True)
def eng(d):
    e=d.y_pred-d.y_true
    return np.sqrt(pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').e2.mean())
print('subset  engines  FULL_better  mean_rmse_M0  mean_rmse_FULL')
for s in ['FD001','FD002','FD003','FD004']:
    m=load(s,'M0'); f=load(s,'FULL')
    if m is None or f is None: continue
    em,ef=eng(m),eng(f); idx=em.index.intersection(ef.index)
    print('%-7s %6d  %5.0f%%       %6.2f        %6.2f'%(s,len(idx),100*(ef.loc[idx]<em.loc[idx]).mean(),em.loc[idx].mean(),ef.loc[idx].mean()))
