import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def met(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return np.sqrt(g.e2.mean()), g.ae.mean(), g.s.mean(), g.b.mean()
for subset in ['FD002','FD004']:
    S=[];R=[]
    for k in range(5):
        a=H/'runs'/subset/str(k)/'FULL_202.csv'
        b=H/'runs_v3_DM202_w50'/subset/str(k)/'M0_202.csv'
        if not (a.is_file() and b.is_file()): continue
        S.append(pd.read_csv(b)); R.append(pd.read_csv(a))
    if not S: print(subset,'none'); continue
    s=met(pd.concat(S,ignore_index=True)); r=met(pd.concat(R,ignore_index=True))
    print('seed202 %s DM(w50) %d folds: RMSE %.3f vs FULL %.3f  %+.2f%%  MAE %.3f/%.3f  NASA %.2f/%.2f'%(
        subset,len(S),s[0],r[0],100*(s[0]/r[0]-1),s[1],r[1],s[2],r[2]))