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
    g=H/f'runs3_w50_graph/FD002/2/{arm}_101.csv'
    if not g.is_file(): print(arm,'graph missing'); continue
    rb=met(b); rg=met(g)
    print('%s BASE %.3f -> GRAPH %.3f (%+.1f%%)  NASA %.2f -> %.2f  bias %+.3f -> %+.3f'%(
        arm, rb[0], rg[0], 100*(rg[0]/rb[0]-1), rb[2], rg[2], rb[3], rg[3]))