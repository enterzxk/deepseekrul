"""Observation-span schedule: pooled metrics over all folds (diluted where the w100
model is missing) plus the equal-weight mean over covered folds (the full-coverage
estimate). Writes SCHEDULE.csv / SCHEDULE.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
REPS=5000
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def load(subset, fold, arm):
    a=HERE/'runs'/subset/str(fold)/f'{arm}_101.csv'
    b=HERE/'runs3_w100_w100'/subset/str(fold)/f'{arm}_101.csv'
    if not a.is_file(): return None, None
    return pd.read_csv(a), (pd.read_csv(b) if b.is_file() else None)
def schedule(w50, w100):
    d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
    sel=(d.end_cycle>=100)&d.y_pred_t.notna()
    d.loc[sel.values,'y_pred']=d.loc[sel,'y_pred_t'].values
    return d
def metrics(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return dict(rmse=float(np.sqrt(g.e2.mean())),mae=float(g.ae.mean()),nasa=float(g.s.mean()),bias=float(g.b.mean()),engines=int(len(g))), g
def paired(a,b):
    idx=a.index.intersection(b.index); av=a.loc[idx,'e2'].to_numpy(); bv=b.loc[idx,'e2'].to_numpy()
    rng=np.random.default_rng(99); dr=rng.integers(0,len(idx),(REPS,len(idx)))
    dif=np.sqrt(av[dr].mean(1))-np.sqrt(bv[dr].mean(1))
    return float(np.sqrt(av.mean())-np.sqrt(bv.mean())), float(np.percentile(dif,2.5)), float(np.percentile(dif,97.5))
rows=[]
for subset in ['FD002','FD004']:
    for arm in ['M0','FULL']:
        sch=[]; ref=[]; per=[]
        for k in range(5):
            a,b=load(subset,k,arm)
            if a is None: continue
            if b is None:
                sch.append(a.assign(fold=k)); ref.append(a.assign(fold=k)); continue
            s=schedule(a,b)
            per.append((k, metrics(a)[0]['rmse'], metrics(s)[0]['rmse']))
            sch.append(s.assign(fold=k)); ref.append(a.assign(fold=k))
        if not sch: continue
        S=pd.concat(sch,ignore_index=True); R=pd.concat(ref,ignore_index=True)
        ms,_=metrics(S); mr,_=metrics(R)
        d,lo,hi=paired(metrics(S)[1],metrics(R)[1])
        rel_cov = 100*(np.mean([p[2] for p in per])/np.mean([p[1] for p in per])-1) if per else float('nan')
        rows.append(dict(subset=subset,arm=arm,covered_folds=len(per),
            sched_rmse=ms['rmse'],sched_mae=ms['mae'],sched_nasa=ms['nasa'],sched_bias=ms['bias'],
            ref_rmse=mr['rmse'],delta=d,ci_lo=lo,ci_hi=hi,
            rel_all=100*(ms['rmse']/mr['rmse']-1),rel_covered=rel_cov,engines=ms['engines']))
df=pd.DataFrame(rows)
if len(df):
    df.to_csv(HERE/'SCHEDULE.csv',index=False)
    (HERE/'SCHEDULE.json').write_text(df.to_json(orient='records',indent=2))
print(df.round(3).to_string(index=False))