"""Which engines drive the FULL-vs-M0 difference on FD004 (and FD002)?"""
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
for subset in ['FD002','FD004']:
    m=load(subset,'M0'); f=load(subset,'FULL')
    if m is None or f is None: continue
    def eng(d):
        e=d.y_pred-d.y_true
        return pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').e2.mean()
    em=eng(m); ef=eng(f)
    idx=em.index.intersection(ef.index)
    dm=np.sqrt(em.loc[idx]); df_=np.sqrt(ef.loc[idx])
    delta=df_-dm
    print('==',subset,'engines',len(idx))
    print('   FULL better on %d (%.0f%%), worse on %d'%((delta<0).sum(),100*(delta<0).mean(),(delta>0).sum()))
    print('   mean per-engine rmse: M0 %.2f FULL %.2f'%(dm.mean(),df_.mean()))
    top=delta.sort_values()
    print('   largest improvements (FULL-M0):', ', '.join('%d:%+.1f'%(u,v) for u,v in top.head(5).items()))
    print('   largest regressions:', ', '.join('%d:%+.1f'%(u,v) for u,v in top.tail(5).items()))
    # contribution of the improved/worsened halves to the pooled change
    print('   sum of delta over engines: %+.1f ; pooled change in RMSE: %+.3f'%(delta.sum(), np.sqrt((ef.loc[idx]**2).mean())-np.sqrt((em.loc[idx]**2).mean()) if False else np.sqrt((ef.loc[idx]).mean())-np.sqrt((em.loc[idx]).mean())))
