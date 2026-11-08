import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
w50=pd.read_csv(HERE/'runs'/'FD002'/'2'/'M0_101.csv')
w100=pd.read_csv(HERE/'runs3_w100'/'FD002'/'2'/'M0_101.csv')
print('w50 all windows: %.3f n=%d'%(rmse(w50),len(w50)))
print('w100 windows:    %.3f n=%d'%(rmse(w100),len(w100)))
sub=w50[w50.end_cycle>=100]
print('w50 restricted to end_cycle>=100: %.3f n=%d'%(rmse(sub),len(sub)))
m=w50.merge(w100,on=['unit','end_cycle'],suffixes=('_50','_100'))
print('matched rows:',len(m))
print('matched w50: %.3f   w100: %.3f'%(rmse(m.rename(columns={'y_pred_50':'y_pred','y_true_50':'y_true'})),
      rmse(m.rename(columns={'y_pred_100':'y_pred','y_true_100':'y_true'}))))