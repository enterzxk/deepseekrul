import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
late=pd.read_csv(HERE/'runs_late_c100_w50'/'FD002'/'2'/'M0_101.csv')
w50=pd.read_csv(HERE/'runs'/'FD002'/'2'/'M0_101.csv')
w100=pd.read_csv(HERE/'runs3_w100_w100'/'FD002'/'2'/'M0_101.csv')
print('late-specialist w50 (trained+eval on age>=100): %.3f  n=%d'%(rmse(late),len(late)))
print('generalist w50 on same windows:               %.3f'%rmse(w50[w50.end_cycle>=100]))
print('w100 on same windows:                         %.3f'%rmse(w100))
m=late.merge(w50[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],suffixes=('','_g'))
m=m.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],suffixes=('','_w100'))
print('matched n=%d'%len(m))
print('  spec-w50 %.3f | gen-w50 %.3f | w100 %.3f'%(rmse(m),rmse(m.assign(y_pred=m.y_pred_g)),rmse(m.assign(y_pred=m.y_pred_w100))))