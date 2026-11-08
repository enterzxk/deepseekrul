"""Apply outer-val LOFO fusion weights to the official test predictions."""
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105')
W={'FD002':np.array([0.375,0.200,0.420])}
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
for subset in ['FD002']:
    y=np.loadtxt(f'/home/user/rulb/data/CMAPSS/RUL_{subset}.txt').reshape(-1).clip(0,125)
    P=[]
    for arm in ['M0','OPS','FULL']:
        preds=[pd.read_csv(HERE/'official_fullbudget'/subset/f'{arm}_{s}.csv').y_pred.to_numpy() for s in [101,202,303]]
        P.append(np.mean(preds,axis=0))
    P=np.stack(P,axis=1)
    for name,w in [('equal',np.ones(3)/3),('lofo_weights',W[subset]),('OPS_only',np.array([0,1,0.0]))]:
        e=P@w-y
        print('%-12s RMSE %.3f MAE %.3f NASA %.1f bias %+.2f'%(name,np.sqrt((e**2).mean()),np.abs(e).mean(),nasa(e).mean(),e.mean()))
