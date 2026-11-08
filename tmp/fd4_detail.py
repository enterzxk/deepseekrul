import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
for k in [0,1]:
    a=HERE/'runs'/'FD004'/str(k)/'M0_101.csv'; b=HERE/'runs3_w100_w100'/'FD004'/str(k)/'M0_101.csv'
    w50=pd.read_csv(a); w100=pd.read_csv(b)
    late=w50[w50.end_cycle>=100]
    m=late.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],suffixes=('','_o'))
    print('FD004 f%d: total %d, late %d (%.0f%%); late w50 %.3f -> w100 %.3f (%+.1f%%)'%(
        k,len(w50),len(late),100*len(late)/len(w50),rmse(m),rmse(m.assign(y_pred=m.y_pred_o)),
        100*(rmse(m.assign(y_pred=m.y_pred_o))/rmse(m)-1)))