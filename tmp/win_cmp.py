import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(p):
    d=pd.read_csv(p); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean())), len(d)
for tag in ['runs3_w100','runs3_w150']:
    for arm in ['M0','FULL']:
        p=HERE/tag/'FD002'/'2'/f'{arm}_101.csv'
        if p.is_file():
            r,n=rmse(p); print('%s %s %.3f (n=%d)'%(tag,arm,r,n))
        else:
            print('%s %s pending'%(tag,arm))
print('baseline w50 fold2: M0 13.732  FULL 12.751')