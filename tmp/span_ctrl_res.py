import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
ctrl=pd.read_csv(HERE/'runs_span100_s2'/'FD002'/'2'/'M0_101.csv')
w50=pd.read_csv(HERE/'runs'/'FD002'/'2'/'M0_101.csv')
w100=pd.read_csv(HERE/'runs3_w100_w100'/'FD002'/'2'/'M0_101.csv')
late=w50[w50.end_cycle>=100]
m=late.merge(ctrl[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],suffixes=('','_c'))
m=m.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],suffixes=('','_w100'))
print('late windows n=%d'%len(m))
print('  generalist w50      %.3f'%rmse(m))
print('  span100 / 50 pos    %.3f  (same params as w50)'%rmse(m.assign(y_pred=m.y_pred_c)))
print('  w100 / 100 pos      %.3f  (+3200 params)'%rmse(m.assign(y_pred=m.y_pred_w100)))
print('  late-spec w50       %.3f'%10.649)