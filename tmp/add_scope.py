import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
TAU=50
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def met(d,col):
    e=d[col]-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean())),float(g.ae.mean()),float(g.s.mean()),float(g.b.mean()),len(g)
rows=[]
for subset in ['FD001','FD003']:
    S=[];R=[]
    for k in range(5):
        b=H/'runs_v3_DM_w50'/subset/str(k)/'M0_101.csv'; c=H/'runs_v3_DM_w100'/subset/str(k)/'M0_101.csv'
        a=H/'runs'/subset/str(k)/'FULL_101.csv'
        if not (b.is_file() and c.is_file() and a.is_file()): continue
        w50=pd.read_csv(b); w100=pd.read_csv(c)
        d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
        late=(d.end_cycle>=100)&d.y_pred_t.notna()
        s=np.where(late,d.y_pred_t,d.y_pred)
        dis=(d.y_pred_t-d.y_pred).abs()>TAU
        d['y']=np.where(late&dis,d.y_pred,s)
        S.append(d); R.append(pd.read_csv(a))
    if not S: continue
    ms=met(pd.concat(S,ignore_index=True),'y'); mr=met(pd.concat(R,ignore_index=True),'y_pred')
    rows.append(dict(subset=subset,arm='MIXUP_SPAN',n_seeds=1,rmse_mean=round(ms[0],4),rmse_sd=0.0,mae=round(ms[1],4),
        nasa=round(ms[2],4),bias=round(ms[3],4),engines=ms[4],windows=0,reference='FULL',
        delta_rmse=round(ms[0]-mr[0],4),ci_lo='',ci_hi='',gain=round(ms[0]/mr[0]-1,4)))
    print('%s MIXUP_SPAN %.3f vs FULL %.3f  %+.2f%%'%(subset,ms[0],mr[0],100*(ms[0]/mr[0]-1)))
r=pd.read_csv(H/'RESULTS.csv')
r=r[~((r.arm=='MIXUP_SPAN') & (r.subset.isin(['FD001','FD003'])))]
out=pd.concat([r,pd.DataFrame(rows)],ignore_index=True)
out.to_csv(H/'RESULTS.csv',index=False); print('rows',len(out))