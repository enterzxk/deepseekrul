"""Fair dual-baseline eval: pool only over folds where every variant exists."""
import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
TAU=50
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def met(d,col='y_pred'):
    e=d[col]-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return np.sqrt(g.e2.mean()), g.ae.mean(), g.s.mean(), g.b.mean(), len(g)
def sched(w50,w100):
    d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
    late=(d.end_cycle>=100)&d.y_pred_t.notna()
    s=np.where(late,d.y_pred_t,d.y_pred)
    dis=(d.y_pred_t-d.y_pred).abs()>TAU
    d['y']=np.where(late&dis,d.y_pred,s)
    return d
for subset in ['FD002','FD004']:
    avail={}
    for k in range(5):
        a=H/'runs'/subset/str(k); b=H/'runs_v3_DM_w50'/subset/str(k); c=H/'runs_v3_DM_w100'/subset/str(k)
        if not ((a/'M0_101.csv').is_file() and (b/'M0_101.csv').is_file()): continue
        rec={'pe':pd.read_csv(a/'M0_101.csv'),'dm':pd.read_csv(b/'M0_101.csv')}
        if (a/'FULL_101.csv').is_file(): rec['fu']=pd.read_csv(a/'FULL_101.csv')
        if (c/'M0_101.csv').is_file(): rec['sp']=sched(pd.read_csv(b/'M0_101.csv'),pd.read_csv(c/'M0_101.csv'))
        avail[k]=rec
    if not avail: print('%s: no DM results yet'%subset); continue
    keys=[k for k,v in avail.items() if 'fu' in v]
    print('=== %s === folds with PE+DM: %s'%(subset,sorted(avail)))
    def show(name,col):
        d=pd.concat([avail[k][name] for k in keys],ignore_index=True)
        r=met(d,col); print('  %-12s RMSE %.3f  MAE %.3f  NASA %.2f  bias %+.3f  engines %d'%((name.upper(),)+r))
        return r
    rpe=show('pe','y_pred'); rfu=show('fu','y_pred'); rdm=show('dm','y_pred')
    if all('sp' in avail[k] for k in keys): show('sp','y')
    print('  DM vs PE %+.1f%%   DM vs FULL %+.1f%%'%(100*(rdm[0]/rpe[0]-1),100*(rdm[0]/rfu[0]-1)))
    print()