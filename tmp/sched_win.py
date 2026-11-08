import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def eng(d):
    e=d.y_pred-d.y_true
    return np.sqrt(pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').e2.mean())
for k in [0,2]:
    w50=pd.read_csv(HERE/'runs'/'FD002'/str(k)/'M0_101.csv')
    p=HERE/'runs3_w100_w100'/'FD002'/str(k)/'M0_101.csv'
    if not p.is_file(): print('fold',k,'pending'); continue
    w100=pd.read_csv(p)
    hyb=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
    sel=(hyb.end_cycle>=100)&hyb.y_pred_t.notna()
    hyb.loc[sel.values,'y_pred']=hyb.loc[sel,'y_pred_t'].values
    a=eng(w50); b=eng(hyb); idx=a.index.intersection(b.index)
    win=(b.loc[idx]<a.loc[idx]).mean()
    print('fold %d: schedule better on %.0f%% of %d engines; mean per-engine RMSE %.2f -> %.2f'%(k,100*win,len(idx),a.loc[idx].mean(),b.loc[idx].mean()))