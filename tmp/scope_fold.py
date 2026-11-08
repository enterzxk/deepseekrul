from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(p):
    d=pd.read_csv(p); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
for subset in ['FD001','FD003']:
    out=[]
    for k in range(5):
        m=HERE/'runs'/subset/str(k)/'M0_101.csv'; f=HERE/'runs'/subset/str(k)/'FULL_101.csv'
        if m.is_file() and f.is_file():
            out.append('f%d M0 %.2f FULL %.2f (%+.1f%%)'%(k,rmse(m),rmse(f),100*(rmse(f)/rmse(m)-1)))
    print(subset, '|', '  '.join(out) if out else 'pending')
