import pandas as pd, numpy as np
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def met(f):
    d=pd.read_csv(f); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return np.sqrt(g.e2.mean()), g.ae.mean(), g.s.mean(), g.b.mean()
for arm in ['M0','FULL']:
    b=H/f'runs/FD002/2/{arm}_101.csv'
    print('%-4s BASE RMSE %.3f MAE %.3f NASA %.2f bias %+.3f'%(arm,*met(b)))
    for tag in ['A','B','AB','BASE']:
        f=H/f'runs_v3_{tag}/FD002/2/{arm}_101.csv'
        if f.is_file():
            r=met(f); rb=met(b)
            print('%-4s %-5s RMSE %.3f (%+.1f%%) MAE %.3f NASA %.2f bias %+.3f'%(arm,tag,r[0],100*(r[0]/rb[0]-1),r[1],r[2],r[3]))