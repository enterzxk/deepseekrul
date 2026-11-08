from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(p):
    d=pd.read_csv(p); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
for subset in ['FD002','FD004']:
    print('==',subset)
    for arm in ['M0','OPS','FULL','NORM']:
        vals=[]
        for k in range(5):
            p=HERE/'runs'/subset/str(k)/f'{arm}_101.csv'
            vals.append('%.2f'%rmse(p) if p.is_file() else '--')
        print('  %-5s %s'%(arm,' '.join(vals)))
