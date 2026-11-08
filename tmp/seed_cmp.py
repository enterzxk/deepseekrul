from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(p):
    d=pd.read_csv(p); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
for seed in [101,202,303]:
    line=[]
    for arm in ['M0','OPS','FULL','NORM']:
        vals=[]
        for k in range(5):
            p=HERE/'runs/FD002'/str(k)/f'{arm}_{seed}.csv'
            if p.is_file(): vals.append(rmse(p))
        if vals: line.append('%s %.3f(n=%d)'%(arm,np.mean(vals),len(vals)))
    print('seed',seed,':',' '.join(line))
