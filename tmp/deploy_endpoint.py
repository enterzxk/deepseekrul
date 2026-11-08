"""Deployment-matched endpoint: one window per engine at a pseudo-truncation cycle."""
import numpy as np, pandas as pd
from pathlib import Path
def load(subset,arm):
    b=[]
    for k in range(5):
        p=Path(f'runs/{subset}/{k}/{arm}_101.csv')
        if p.is_file(): b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b,ignore_index=True) if b else None
def pooled(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs()}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
rng=np.random.default_rng(20261105)
for subset in ['FD002']:
    rul=np.loadtxt(f'/home/user/rulb/data/CMAPSS/RUL_{subset}.txt').reshape(-1)
    print('  test RUL: n=%d mean %.1f min %.0f max %.0f'%(len(rul),rul.mean(),rul.min(),rul.max()))
    arms={a:load(subset,a) for a in ['M0','OPS','FULL']}
    arms={a:d for a,d in arms.items() if d is not None}
    ref=arms['M0']
    T=ref.groupby('unit').end_cycle.max()
    units=T.index.to_numpy()
    draws=rng.choice(rul,size=len(units),replace=True)
    cut={u:int(T[u])-int(d) for u,d in zip(units,draws)}
    sel=[]
    for u in units:
        d=ref[ref.unit==u]
        sub=d[d.end_cycle<=cut[u]]
        if len(sub)==0: continue
        sel.append(sub.sort_values('end_cycle').iloc[-1][['unit','end_cycle']])
    key=pd.DataFrame(sel)
    print('==',subset,'engines',len(key),'mean cutoff %.0f  mean true RUL %.0f'%(key.end_cycle.mean(),
        key.merge(ref,on=['unit','end_cycle']).y_true.mean()))
    preds={}
    for a,d in arms.items():
        m=key.merge(d,on=['unit','end_cycle'],how='inner'); preds[a]=m
        print('  %-5s RMSE %.3f  MAE %.3f  bias %+.2f  n=%d'%(a,pooled(m),(m.y_pred-m.y_true).abs().mean(),(m.y_pred-m.y_true).mean(),len(m)))
    if len(preds)==3:
        base=preds['M0']
        for combo in [('M0','FULL'),('M0','OPS'),('M0','OPS','FULL')]:
            pr=np.mean([preds[a].y_pred.to_numpy() for a in combo],axis=0)
            print('  blend %-14s %.3f'%('+'.join(combo),pooled(base.assign(y_pred=pr))))
