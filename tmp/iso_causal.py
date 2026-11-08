"""Causal (prefix-only) monotone smoothing: at time t use only windows ending <= t."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def load(subset, arm, seed=101):
    b=[]
    for k in range(5):
        p=HERE/'runs'/subset/str(k)/f'{arm}_{seed}.csv'
        if not p.is_file(): return None
        b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b, ignore_index=True)
def pooled(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
def causal(d):
    out=[]
    for (f,u),g in d.groupby(['fold','unit']):
        g=g.sort_values('end_cycle').reset_index(drop=True)
        pred=g.y_pred.to_numpy(float); cyc=g.end_cycle.to_numpy(float); res=np.empty_like(pred)
        for i in range(len(g)):
            if i<2:
                res[i]=pred[i]; continue
            ir=IsotonicRegression(increasing=False, out_of_bounds='clip')
            ir.fit(cyc[:i+1], pred[:i+1])
            res[i]=float(ir.predict([cyc[i]])[0])
        out.append(g.assign(y_pred=res))
    return pd.concat(out, ignore_index=True)
for subset in ['FD002','FD004']:
    for arm in ['M0','FULL']:
        d=load(subset,arm)
        if d is None: continue
        raw=pooled(d); ca=pooled(causal(d))
        print('%s %-4s raw %.3f  causal-iso %.3f  (%+.2f%%)'%(subset,arm,raw,ca,100*(ca/raw-1)))