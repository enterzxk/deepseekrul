import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
def load(p):
    return pd.read_csv(p) if Path(p).is_file() else None
w50m=load(HERE/'runs'/'FD002'/'2'/'M0_101.csv'); w50f=load(HERE/'runs'/'FD002'/'2'/'FULL_101.csv')
w100m=load(HERE/'runs3_w100_w100'/'FD002'/'2'/'M0_101.csv'); w100f=load(HERE/'runs3_w100_w100'/'FD002'/'2'/'FULL_101.csv')
w150m=load(HERE/'runs3_w150_w150'/'FD002'/'2'/'M0_101.csv'); w150f=load(HERE/'runs3_w150_w150'/'FD002'/'2'/'FULL_101.csv')
for name,d in [('w50 M0',w50m),('w50 FULL',w50f),('w100 M0',w100m),('w100 FULL',w100f),('w150 M0',w150m),('w150 FULL',w150f)]:
    if d is not None: print('%-10s %.3f  n=%d'%(name,rmse(d),len(d)))
    else: print('%-10s pending'%name)
base=w50m
for cut in [100,150]:
    sub=base[base.end_cycle>=cut]
    line='end>=%d: w50 %.3f'%(cut,rmse(sub))
    for name,d in [('w100',w100m),('w150',w150m)]:
        if d is None: continue
        mm=sub.merge(d[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],suffixes=('','_o'))
        line+='  %s %.3f (n=%d)'%(name,rmse(mm.assign(y_pred=mm.y_pred_o)),len(mm))
    print(line)