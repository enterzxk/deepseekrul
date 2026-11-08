"""Stratified (by subset) engine-clustered bootstrap for the schedule contrast."""
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
REPS=5000
def load(subset, fold, arm):
    a=HERE/'runs'/subset/str(fold)/f'{arm}_101.csv'
    b=HERE/'runs3_w100_w100'/subset/str(fold)/f'{arm}_101.csv'
    if not a.is_file() or not b.is_file(): return None, None
    return pd.read_csv(a), pd.read_csv(b)
def sched(w50, w100):
    d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
    sel=(d.end_cycle>=100)&d.y_pred_t.notna()
    d.loc[sel.values,'y_pred']=d.loc[sel,'y_pred_t'].values
    return d
def eng(d):
    e=d.y_pred-d.y_true
    return pd.DataFrame({'u':d.unit.astype(str)+'_'+d.fold.astype(str) if 'fold' in d else d.unit.astype(str),'e2':e**2}).groupby('u').mean()
parts={}
for subset in ['FD002','FD004']:
    S=[]; R=[]
    for k in range(5):
        a,b=load(subset,k,'FULL')
        if a is None: continue
        S.append(sched(a,b).assign(fold=k)); R.append(a.assign(fold=k))
    if S: parts[subset]=(eng(pd.concat(S,ignore_index=True)), eng(pd.concat(R,ignore_index=True)))
for arm in ['FULL']:
    if not parts: continue
    rng=np.random.default_rng(7)
    diffs=[]
    for _ in range(REPS):
        num=0.0; den=0.0; n=0
        for subset,(s,r) in parts.items():
            idx=r.index.intersection(s.index)
            pick=rng.integers(0,len(idx),(len(idx),))
            sv=s.loc[idx,'e2'].to_numpy()[pick]; rv=r.loc[idx,'e2'].to_numpy()[pick]
            num+=sv.sum(); den+=rv.sum(); n+=len(pick)
        diffs.append(np.sqrt(num/n)-np.sqrt(den/n))
    s_all=np.sqrt(sum(s.e2.sum() for s,_ in parts.values())/sum(len(s) for s,_ in parts.values()))
    r_all=np.sqrt(sum(r.e2.sum() for _,r in parts.values())/sum(len(r) for _,r in parts.values()))
    print('stratified %s: sched %.3f ref %.3f delta %+.3f  95%% CI [%+.3f, %+.3f]  rel %+.2f%%'%(
        arm,s_all,r_all,s_all-r_all,np.percentile(diffs,2.5),np.percentile(diffs,97.5),100*(s_all/r_all-1)))