"""Stratified bootstrap for DM + span + guard vs the FULL baseline (seed 101)."""
import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
TAU=50; REPS=5000
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def eng(d,col):
    e=d[col]-d.y_true
    return pd.DataFrame({'u':d.unit.astype(str),'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
def build(subset):
    S=[];R=[]
    for k in range(5):
        a=H/'runs'/subset/str(k); b=H/'runs_v3_DM_w50'/subset/str(k); c=H/'runs_v3_DM_w100'/subset/str(k)
        if not ((a/'FULL_101.csv').is_file() and (b/'M0_101.csv').is_file() and (c/'M0_101.csv').is_file()): continue
        w50=pd.read_csv(b/'M0_101.csv'); w100=pd.read_csv(c/'M0_101.csv')
        d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
        late=(d.end_cycle>=100)&d.y_pred_t.notna()
        s=np.where(late,d.y_pred_t,d.y_pred)
        dis=(d.y_pred_t-d.y_pred).abs()>TAU
        d['y']=np.where(late&dis,d.y_pred,s)
        S.append(d.assign(fold=k)); R.append(pd.read_csv(a/'FULL_101.csv').assign(fold=k))
    return (eng(pd.concat(S,ignore_index=True),'y'), eng(pd.concat(R,ignore_index=True),'y_pred')) if S else None
parts={}
for subset in ['FD002','FD004']:
    r=build(subset)
    if r is None: print(subset,'incomplete'); continue
    parts[subset]=r
    s,rf=r
    print('%s: DM+span+guard RMSE %.3f  FULL %.3f  %+.2f%%  MAE %.3f/%.3f  NASA %.2f/%.2f'%(
        subset,np.sqrt(s.e2.mean()),np.sqrt(rf.e2.mean()),100*(np.sqrt(s.e2.mean())/np.sqrt(rf.e2.mean())-1),
        s.ae.mean(),rf.ae.mean(),s.s.mean(),rf.s.mean()))
if len(parts)==2:
    rng=np.random.default_rng(23); diffs=[]
    for _ in range(REPS):
        num=den=0.0; n=0
        for s,rf in parts.values():
            idx=rf.index.intersection(s.index); pick=rng.integers(0,len(idx),(len(idx),))
            num+=s.loc[idx,'e2'].to_numpy()[pick].sum(); den+=rf.loc[idx,'e2'].to_numpy()[pick].sum(); n+=len(pick)
        diffs.append(np.sqrt(num/n)-np.sqrt(den/n))
    sa=np.sqrt(sum(s.e2.sum() for s,_ in parts.values())/sum(len(s) for s,_ in parts.values()))
    ra=np.sqrt(sum(r.e2.sum() for _,r in parts.values())/sum(len(r) for _,r in parts.values()))
    print('stratified: sched %.3f ref %.3f delta %+.3f CI [%+.3f, %+.3f] rel %+.2f%%'%(
        sa,ra,sa-ra,np.percentile(diffs,2.5),np.percentile(diffs,97.5),100*(sa/ra-1)))