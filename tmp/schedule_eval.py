"""Evaluate the observation-span schedule across folds with engine-clustered bootstrap."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
REPS=5000
def load(subset, fold, arm, span):
    if span==50: p=HERE/'runs'/subset/str(fold)/f'{arm}_101.csv'
    elif span==100: p=HERE/'runs3_w100_w100'/subset/str(fold)/f'{arm}_101.csv'
    elif span==150: p=HERE/'runs3_w150_w150'/subset/str(fold)/f'{arm}_101.csv'
    return pd.read_csv(p) if p.is_file() else None
def schedule(subset, folds, arm):
    parts=[]
    for k in folds:
        base=load(subset,k,arm,50)
        if base is None: return None
        out=base.copy().assign(fold=k)
        for span,cut in [(100,100),(150,150)]:
            d=load(subset,k,arm,span)
            if d is None: continue
            mm=out.merge(d[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
            sel=(mm.end_cycle>=cut) & mm.y_pred_t.notna()
            out.loc[sel.values,'y_pred']=mm.loc[sel,'y_pred_t'].values
        parts.append(out)
    return pd.concat(parts,ignore_index=True)
def eng(d):
    e=d.y_pred-d.y_true
    return pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
def pooled(t): return float(np.sqrt(t.e2.mean()))
def paired(a,b):
    idx=a.index.intersection(b.index); av=a.loc[idx,'e2'].to_numpy(); bv=b.loc[idx,'e2'].to_numpy()
    rng=np.random.default_rng(99); dr=rng.integers(0,len(idx),(REPS,len(idx)))
    dif=np.sqrt(av[dr].mean(1))-np.sqrt(bv[dr].mean(1))
    return float(np.sqrt(av.mean())-np.sqrt(bv.mean())), float(np.percentile(dif,2.5)), float(np.percentile(dif,97.5))
for subset in ['FD002','FD004']:
    for arm in ['M0','FULL']:
        s=schedule(subset,range(5),arm)
        b=load(subset,0,arm,50)
        if s is None or b is None:
            print(subset,arm,'incomplete'); continue
        ref=pd.concat([load(subset,k,arm,50).assign(fold=k) for k in range(5)],ignore_index=True)
        d,lo,hi=paired(eng(s),eng(ref))
        print('%s %-4s schedule %.3f  baseline %.3f  delta %+.3f [%+.3f,%+.3f]  rel %+.2f%%'%(
            subset,arm,pooled(eng(s)),pooled(eng(ref)),d,lo,hi,100*(pooled(eng(s))/pooled(eng(ref))-1)))