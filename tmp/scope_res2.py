import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
TAU=50
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def met(d,col='y_pred'):
    e=d[col]-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return np.sqrt(g.e2.mean()), g.ae.mean(), g.s.mean(), g.b.mean()
for subset in ['FD001','FD003']:
    dm=[];sp=[];fu=[];pe=[]
    for k in range(5):
        b=H/'runs_v3_DM_w50'/subset/str(k)/'M0_101.csv'
        c=H/'runs_v3_DM_w100'/subset/str(k)/'M0_101.csv'
        a=H/'runs'/subset/str(k)/'FULL_101.csv'
        p=H/'runs'/subset/str(k)/'M0_101.csv'
        if b.is_file(): dm.append(pd.read_csv(b))
        if a.is_file(): fu.append(pd.read_csv(a))
        if p.is_file(): pe.append(pd.read_csv(p))
        if b.is_file() and c.is_file():
            w50=pd.read_csv(b); w100=pd.read_csv(c)
            d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
            late=(d.end_cycle>=100)&d.y_pred_t.notna()
            s=np.where(late,d.y_pred_t,d.y_pred)
            dis=(d.y_pred_t-d.y_pred).abs()>TAU
            d['y']=np.where(late&dis,d.y_pred,s)
            sp.append(d)
    if not dm: print(subset,'none'); continue
    rf=met(pd.concat(fu,ignore_index=True)) if fu else None
    rp=met(pd.concat(pe,ignore_index=True)) if pe else None
    r1=met(pd.concat(dm,ignore_index=True))
    out='%s: DM(w50) %.3f (%d folds)'%(subset,r1[0],len(dm))
    if rf: out+='  vs FULL %+.2f%%'%(100*(r1[0]/rf[0]-1))
    if rp: out+='  vs PE %+.2f%%'%(100*(r1[0]/rp[0]-1))
    print(out)
    if sp:
        r2=met(pd.concat(sp,ignore_index=True),'y')
        out2='%s: DM+span+guard %.3f (%d folds)'%(subset,r2[0],len(sp))
        if rf: out2+='  vs FULL %+.2f%%'%(100*(r2[0]/rf[0]-1))
        if rp: out2+='  vs PE %+.2f%%'%(100*(r2[0]/rp[0]-1))
        print(out2)