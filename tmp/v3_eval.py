"""Evaluate all v3 screening configs against the 42k baseline."""
import pandas as pd, numpy as np, glob
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def met(f):
    d=pd.read_csv(f); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return np.sqrt(g.e2.mean()), g.ae.mean(), g.s.mean(), g.b.mean()
rows=[]
for f in sorted(glob.glob(str(H/'runs_v3_*'/'FD00*'/'*'/'M0_101.csv'))):
    p=Path(f); tag=p.parents[2].name.replace('runs_v3_',''); subset=p.parents[1].name; fold=p.parents[0].name
    for arm in ['M0','FULL']:
        ff=p.parent/f'{arm}_101.csv'
        if not ff.is_file(): continue
        b=H/'runs'/subset/fold/f'{arm}_101.csv'
        if not b.is_file(): continue
        r=met(ff); rb=met(b)
        rows.append(dict(tag=tag,subset=subset,fold=int(fold),arm=arm,rmse=r[0],rel=100*(r[0]/rb[0]-1),
                         mae=r[1],nasa=r[2],bias=r[3],base_rmse=rb[0],base_nasa=rb[2]))
if rows:
    d=pd.DataFrame(rows); pd.set_option('display.width',200)
    print(d.to_string(index=False,float_format=lambda x:'%.3f'%x))
else:
    print('no v3 results yet')