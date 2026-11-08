"""Dual-baseline evaluation: PE (plain backbone) and FULL (prior), plus the mixup variant."""
import numpy as np, pandas as pd, glob
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
TAU=50
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def load(f): return pd.read_csv(f)
def sched(w50,w100,guard=True):
    d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
    late=(d.end_cycle>=100)&d.y_pred_t.notna()
    s=np.where(late,d.y_pred_t,d.y_pred)
    if guard:
        dis=(d.y_pred_t-d.y_pred).abs()>TAU
        d['y']=np.where(late&dis,d.y_pred,s)
    else:
        d['y']=s
    return d
def met(d,col='y_pred'):
    e=d[col]-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return np.sqrt(g.e2.mean()), g.ae.mean(), g.s.mean(), g.b.mean()
for subset in ['FD002','FD004']:
    pe=[]; fu=[]; dm=[]; sp=[]
    for k in range(5):
        a=H/'runs'/subset/str(k); b=H/'runs_v3_DM_w50'/subset/str(k); c=H/'runs_v3_DM_w100'/subset/str(k)
        if (a/'M0_101.csv').is_file(): pe.append(load(a/'M0_101.csv'))
        if (a/'FULL_101.csv').is_file(): fu.append(load(a/'FULL_101.csv'))
        if (b/'M0_101.csv').is_file(): dm.append(load(b/'M0_101.csv'))
        if (b/'M0_101.csv').is_file() and (c/'M0_101.csv').is_file():
            sp.append(sched(load(b/'M0_101.csv'),load(c/'M0_101.csv')))
    def show(name,lst,col='y_pred'):
        if not lst: print('%-14s missing'%name); return None
        d=pd.concat(lst,ignore_index=True); r=met(d,col)
        print('%-14s folds %d  RMSE %.3f  MAE %.3f  NASA %.2f  bias %+.3f'%(name,len(lst),*r))
        return r
    print('=== %s ==='%subset)
    rpe=show('PE',pe); rfu=show('FULL',fu); rdm=show('DM(w50)',dm)
    rsp=show('DM+span+guard',sp,'y')
    if rpe and rdm: print('  DM vs PE: %+.1f%%   DM vs FULL: %+.1f%%'%(100*(rdm[0]/rpe[0]-1),100*(rdm[0]/rfu[0]-1) if rfu else 0))
    if rsp and rpe: print('  DMspan vs PE: %+.1f%%   vs FULL: %+.1f%%'%(100*(rsp[0]/rpe[0]-1),100*(rsp[0]/rfu[0]-1) if rfu else 0))
    print()