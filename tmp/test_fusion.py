"""Official-test performance of the fixed equal-weight fusion (no test-based tuning)."""
import json
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105')
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
for subset in ['FD001','FD002','FD003','FD004']:
    y=np.loadtxt(f'/home/user/rulb/data/CMAPSS/RUL_{subset}.txt').reshape(-1).clip(0,125)
    ens={}
    for arm in ['M0','OPS','FULL']:
        preds=[pd.read_csv(HERE/'official_fullbudget'/subset/f'{arm}_{s}.csv').y_pred.to_numpy() for s in [101,202,303]]
        ens[arm]=np.mean(preds,axis=0)
    print('==',subset)
    for arm,v in ens.items():
        e=v-y
        print('   %-5s RMSE %.3f MAE %.3f NASA %.1f bias %+.2f'%(arm,np.sqrt((e**2).mean()),np.abs(e).mean(),nasa(e).mean(),e.mean()))
    for combo in [('M0','FULL'),('M0','OPS','FULL')]:
        v=np.mean([ens[a] for a in combo],axis=0); e=v-y
        print('   fuse %-12s RMSE %.3f MAE %.3f NASA %.1f bias %+.2f'%('+'.join(combo),np.sqrt((e**2).mean()),np.abs(e).mean(),nasa(e).mean(),e.mean()))
