import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
def eng(d):
    e=d.y_pred-d.y_true
    return np.sqrt(pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').e2.mean())
for subset in ['FD002','FD004']:
    for arm in ['M0','FULL']:
        for k in range(5):
            a=HERE/'runs'/subset/str(k)/f'{arm}_101.csv'; b=HERE/'runs3_w100_w100'/subset/str(k)/f'{arm}_101.csv'
            if not b.is_file(): continue
            w50=pd.read_csv(a); w100=pd.read_csv(b)
            hyb=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
            sel=(hyb.end_cycle>=100)&hyb.y_pred_t.notna()
            hyb.loc[sel.values,'y_pred']=hyb.loc[sel,'y_pred_t'].values
            e1=eng(w50); e2=eng(hyb); idx=e1.index.intersection(e2.index)
            print('%s %-4s fold %d: w50 %.3f -> sched %.3f (%+.1f%%)  win %.0f%%'%(subset,arm,k,rmse(w50),rmse(hyb),100*(rmse(hyb)/rmse(w50)-1),100*(e2.loc[idx]<e1.loc[idx]).mean()))