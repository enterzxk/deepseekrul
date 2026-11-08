"""Two-seed schedule analysis: four metrics, stratified bootstrap, per-subset gains."""
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
REPS=5000
SEEDS=[101,202]
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def load(subset, fold, arm, seed):
    a=HERE/'runs'/subset/str(fold)/f'{arm}_{seed}.csv'
    b=HERE/'runs3_w100_w100'/subset/str(fold)/f'{arm}_{seed}.csv'
    if not a.is_file() or not b.is_file(): return None
    w50=pd.read_csv(a); w100=pd.read_csv(b)
    d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
    sel=(d.end_cycle>=100)&d.y_pred_t.notna()
    d.loc[sel.values,'y_pred']=d.loc[sel,'y_pred_t'].values
    return d
def met(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit.astype(str)+'_'+d.seed.astype(str),'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return dict(rmse=float(np.sqrt(g.e2.mean())),mae=float(g.ae.mean()),nasa=float(g.s.mean()),bias=float(g.b.mean())), g
print('=== per-seed schedule gains (FULL arm) ===')
for seed in SEEDS:
    for subset in ['FD002','FD004']:
        S=[load(subset,k,'FULL',seed) for k in range(5)]
        if any(x is None for x in S): print(seed,subset,'incomplete'); continue
        S=pd.concat([s.assign(fold=k,seed=seed) for k,s in enumerate(S)],ignore_index=True)
        R=pd.concat([pd.read_csv(HERE/'runs'/subset/str(k)/f'FULL_{seed}.csv').assign(fold=k,seed=seed) for k in range(5)],ignore_index=True)
        ms,_=met(S); mr,_=met(R)
        print('seed %d %s FULL: sched %.3f ref %.3f  %+.2f%%'%(seed,subset,ms['rmse'],mr['rmse'],100*(ms['rmse']/mr['rmse']-1)))
print()
print('=== pooled over both seeds ===')
parts={}
for subset in ['FD002','FD004']:
    S=[]; R=[]
    for seed in SEEDS:
        for k in range(5):
            s=load(subset,k,'FULL',seed)
            if s is None: continue
            S.append(s.assign(fold=k,seed=seed))
            R.append(pd.read_csv(HERE/'runs'/subset/str(k)/f'FULL_{seed}.csv').assign(fold=k,seed=seed))
    S=pd.concat(S,ignore_index=True); R=pd.concat(R,ignore_index=True)
    parts[subset]=(met(S)[1],met(R)[1])
    ms=met(S)[0]; mr=met(R)[0]
    print('%s FULL: sched RMSE %.3f MAE %.3f NASA %.3f bias %+.3f | ref RMSE %.3f MAE %.3f NASA %.3f bias %+.3f | %+.2f%%'%(
        subset,ms['rmse'],ms['mae'],ms['nasa'],ms['bias'],mr['rmse'],mr['mae'],mr['nasa'],mr['bias'],100*(ms['rmse']/mr['rmse']-1)))
rng=np.random.default_rng(11); diffs=[]
for _ in range(REPS):
    num=0.0; den=0.0; n=0
    for subset,(s,r) in parts.items():
        idx=r.index.intersection(s.index)
        pick=rng.integers(0,len(idx),(len(idx),))
        sv=s.loc[idx,'e2'].to_numpy()[pick]; rv=r.loc[idx,'e2'].to_numpy()[pick]
        num+=sv.sum(); den+=rv.sum(); n+=len(pick)
    diffs.append(np.sqrt(num/n)-np.sqrt(den/n))
sa=np.sqrt(sum(s.e2.sum() for s,_ in parts.values())/sum(len(s) for s,_ in parts.values()))
ra=np.sqrt(sum(r.e2.sum() for _,r in parts.values())/sum(len(r) for _,r in parts.values()))
print()
print('stratified 2-seed: sched %.3f ref %.3f delta %+.3f CI [%+.3f, %+.3f] rel %+.2f%%'%(
    sa,ra,sa-ra,np.percentile(diffs,2.5),np.percentile(diffs,97.5),100*(sa/ra-1)))