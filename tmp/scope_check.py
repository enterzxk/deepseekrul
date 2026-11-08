"""FD001/FD003 scope check: FULL vs M0 against the protocol's allowed degradation."""
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
def metrics(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return dict(rmse=float(np.sqrt(g.e2.mean())),mae=float(g.ae.mean()),nasa=float(g.s.mean()),bias=float(g.b.mean()),n=int(len(g)))
for subset in ['FD001','FD003']:
    m=load(subset,'M0'); f=load(subset,'FULL')
    if m is None or f is None:
        print(subset,'incomplete'); continue
    a=metrics(m); b=metrics(f)
    print('== %s (n=%d)'%(subset,a['n']))
    print('   M0   RMSE %.3f MAE %.3f NASA %.2f bias %+.2f'%(a['rmse'],a['mae'],a['nasa'],a['bias']))
    print('   FULL RMSE %.3f MAE %.3f NASA %.2f bias %+.2f'%(b['rmse'],b['mae'],b['nasa'],b['bias']))
    print('   relative: RMSE %+.2f%%  MAE %+.2f%%  NASA %+.2f%%  (protocol allows <= +1.0/1.0/2.0%%)'%(
        100*(b['rmse']/a['rmse']-1),100*(b['mae']/a['mae']-1),100*(b['nasa']/a['nasa']-1)))
