import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
for k in range(5):
    a=HERE/'runs'/'FD002'/str(k)/'M0_101.csv'
    b=HERE/'runs3_w100_w100'/'FD002'/str(k)/'M0_101.csv'
    if not b.is_file(): print('fold',k,'pending'); continue
    w50=pd.read_csv(a); w100=pd.read_csv(b)
    late=w50[w50.end_cycle>=100]
    m=late.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],suffixes=('','_o'))
    hyb=w50.copy().merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
    sel=(hyb.end_cycle>=100)&hyb.y_pred_t.notna()
    hyb.loc[sel.values,'y_pred']=hyb.loc[sel,'y_pred_t'].values
    print('fold %d: w50 %.3f | late w50 %.3f w100 %.3f | schedule %.3f (%+.1f%%)'%(k,rmse(w50),rmse(m),rmse(m.assign(y_pred=m.y_pred_o)),rmse(hyb),100*(rmse(hyb)/rmse(w50)-1)))