import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
w50=pd.read_csv(HERE/'runs'/'FD002'/'2'/'M0_101.csv')
w100=pd.read_csv(HERE/'runs3_w100'/'FD002'/'2'/'M0_101.csv')
m=w50.merge(w100[['unit','end_cycle','y_pred']], on=['unit','end_cycle'], how='left', suffixes=('','_w100'))
print('windows:',len(m),' with w100 pred:',int(m.y_pred_w100.notna().sum()))
hyb=m.copy()
hyb['y_pred']=np.where(hyb.y_pred_w100.notna(), hyb.y_pred_w100, hyb.y_pred)
print('w50 M0           %.3f'%rmse(m))
print('schedule hybrid  %.3f  (%+.2f%%)'%(rmse(hyb),100*(rmse(hyb)/rmse(m)-1)))
sub=m[m.end_cycle>=100]
print('late-only w50 %.3f  w100 %.3f'%(rmse(sub), rmse(sub.assign(y_pred=sub.y_pred_w100))))