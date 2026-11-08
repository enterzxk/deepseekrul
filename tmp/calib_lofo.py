import pandas as pd, numpy as np
from pathlib import Path
base='runs'
def load(subset,arm,seed=101):
    b=[]
    for k in range(5):
        p=Path(f'{base}/{subset}/{k}/{arm}_{seed}.csv')
        if p.is_file(): b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b,ignore_index=True) if b else None
def pooled(x,e):
    gg=pd.DataFrame({'u':x.unit,'e':e**2}).groupby('u').mean(); return float(np.sqrt(gg.e.mean()))
for subset in ['FD002','FD004']:
    d=load(subset,'M0')
    if d is None: continue
    d['err']=d.y_pred-d.y_true
    folds=sorted(d.fold.unique()); print('==',subset,'folds',folds)
    for k in folds:
        tr=d[d.fold!=k].copy(); va=d[d.fold==k]
        w=1.0/tr.groupby('unit').unit.transform('size')
        X=np.c_[np.ones(len(tr)),tr.y_pred.to_numpy()]
        W=w.to_numpy()
        beta=np.linalg.lstsq(X*W[:,None],tr.y_true.to_numpy()*W,rcond=None)[0]
        a,b=beta
        raw=pooled(va,va.err); cal=pooled(va,a+b*va.y_pred-va.y_true)
        print('  fold %d: raw %.3f  calibrated %.3f  (%+.2f%%)  a=%+.2f b=%.3f'%(k,raw,cal,100*(1-cal/raw),a,b))
