"""Cross-subset summary table (M0 vs FULL, four metrics, seed 101)."""
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def load(subset,arm):
    b=[]
    for k in range(5):
        p=HERE/'runs'/subset/str(k)/f'{arm}_101.csv'
        if not p.is_file(): return None
        b.append(pd.read_csv(p))
    return pd.concat(b,ignore_index=True)
def met(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return dict(rmse=float(np.sqrt(g.e2.mean())),mae=float(g.ae.mean()),nasa=float(g.s.mean()),bias=float(g.b.mean()),n=int(len(g)))
rows=[]
for s in ['FD001','FD002','FD003','FD004']:
    m=load(s,'M0'); f=load(s,'FULL')
    if m is None or f is None: continue
    a=met(m); b=met(f)
    rows.append(dict(subset=s,engines=a['n'],
        rmse_M0=a['rmse'],rmse_FULL=b['rmse'],rmse_rel=100*(b['rmse']/a['rmse']-1),
        mae_M0=a['mae'],mae_FULL=b['mae'],mae_rel=100*(b['mae']/a['mae']-1),
        nasa_M0=a['nasa'],nasa_FULL=b['nasa'],nasa_rel=100*(b['nasa']/a['nasa']-1),
        bias_M0=a['bias'],bias_FULL=b['bias']))
df=pd.DataFrame(rows)
df.to_csv(HERE/'paper_tables'/'cross_subset.csv',index=False)
print(df.round(3).to_string(index=False))
