"""Guard verification: per-seed gains + stratified bootstrap at tau=50."""
import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
TAU=50; SEEDS=[101,202]; REPS=5000
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def apply(a,b):
    d=a.merge(b[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
    late=(d.end_cycle>=100)&d.y_pred_t.notna()
    sched=np.where(late,d.y_pred_t,d.y_pred)
    dis=(d.y_pred_t-d.y_pred).abs()>TAU
    d['y_pred_g']=np.where(late&dis,d.y_pred,sched)
    return d
def eng(d,col):
    e=d[col]-d.y_true
    return pd.DataFrame({'u':d.unit.astype(str)+'_'+d.seed.astype(str),'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
parts={}
print('=== per-seed (tau=%d) ==='%TAU)
for subset in ['FD002','FD004']:
    for seed in SEEDS:
        S=[];R=[]
        for k in range(5):
            a=H/'runs'/subset/str(k)/f'FULL_{seed}.csv'; b=H/'runs3_w100_w100'/subset/str(k)/f'FULL_{seed}.csv'
            if not (a.is_file() and b.is_file()): continue
            S.append(apply(pd.read_csv(a),pd.read_csv(b)).assign(fold=k,seed=seed))
            R.append(pd.read_csv(a).assign(fold=k,seed=seed))
        S=pd.concat(S,ignore_index=True); R=pd.concat(R,ignore_index=True)
        gs=eng(S,'y_pred_g'); gr=eng(R,'y_pred')
        print('seed %d %s: RMSE %.3f -> %.3f  %+.2f%%  MAE %.3f->%.3f  NASA %.2f->%.2f'%(
            seed,subset,np.sqrt(gr.e2.mean()),np.sqrt(gs.e2.mean()),100*(np.sqrt(gs.e2.mean())/np.sqrt(gr.e2.mean())-1),
            gr.ae.mean(),gs.ae.mean(),gr.s.mean(),gs.s.mean()))
for subset in ['FD002','FD004']:
    S=[];R=[]
    for seed in SEEDS:
        for k in range(5):
            a=H/'runs'/subset/str(k)/f'FULL_{seed}.csv'; b=H/'runs3_w100_w100'/subset/str(k)/f'FULL_{seed}.csv'
            if not (a.is_file() and b.is_file()): continue
            S.append(apply(pd.read_csv(a),pd.read_csv(b)).assign(fold=k,seed=seed))
            R.append(pd.read_csv(a).assign(fold=k,seed=seed))
    parts[subset]=(eng(pd.concat(S,ignore_index=True),'y_pred_g'),eng(pd.concat(R,ignore_index=True),'y_pred'))
rng=np.random.default_rng(13); diffs=[]
for _ in range(REPS):
    num=den=0.0; n=0
    for subset,(s,r) in parts.items():
        idx=r.index.intersection(s.index)
        pick=rng.integers(0,len(idx),(len(idx),))
        num+=s.loc[idx,'e2'].to_numpy()[pick].sum(); den+=r.loc[idx,'e2'].to_numpy()[pick].sum(); n+=len(pick)
    diffs.append(np.sqrt(num/n)-np.sqrt(den/n))
sa=np.sqrt(sum(s.e2.sum() for s,_ in parts.values())/sum(len(s) for s,_ in parts.values()))
ra=np.sqrt(sum(r.e2.sum() for _,r in parts.values())/sum(len(r) for _,r in parts.values()))
print()
print('stratified tau=%d: sched %.3f ref %.3f delta %+.3f CI [%+.3f, %+.3f] rel %+.2f%%'%(
    TAU,sa,ra,sa-ra,np.percentile(diffs,2.5),np.percentile(diffs,97.5),100*(sa/ra-1)))