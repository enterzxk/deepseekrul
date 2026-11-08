"""Final schedule table: unguarded and guarded (tau=50) for both arms and seeds."""
import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
SEEDS=[101,202]; TAU=50; REPS=5000
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def build(subset,arm,seed,guard):
    S=[];R=[]
    for k in range(5):
        a=H/'runs'/subset/str(k)/f'{arm}_{seed}.csv'
        b=H/'runs3_w100_w100'/subset/str(k)/f'{arm}_{seed}.csv'
        if not (a.is_file() and b.is_file()): continue
        w50=pd.read_csv(a); w100=pd.read_csv(b)
        d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
        late=(d.end_cycle>=100)&d.y_pred_t.notna()
        sched=np.where(late,d.y_pred_t,d.y_pred)
        if guard:
            dis=(d.y_pred_t-d.y_pred).abs()>TAU
            d['y']=np.where(late&dis,d.y_pred,sched)
        else:
            d['y']=sched
        S.append(d.assign(fold=k,seed=seed)); R.append(w50.assign(fold=k,seed=seed))
    if not S: return None
    return pd.concat(S,ignore_index=True), pd.concat(R,ignore_index=True)
def metrics(d,col):
    e=d[col]-d.y_true
    g=pd.DataFrame({'u':d.unit.astype(str)+'_'+d.seed.astype(str),'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return g
rows=[]
for subset in ['FD002','FD004']:
    for arm in ['M0','FULL']:
        for guard in [False,True]:
            S=[];R=[]
            for seed in SEEDS:
                r=build(subset,arm,seed,guard)
                if r is None: continue
                S.append(r[0]); R.append(r[1])
            if not S: continue
            S=pd.concat(S,ignore_index=True); R=pd.concat(R,ignore_index=True)
            gs=metrics(S,'y'); gr=metrics(R,'y_pred')
            rng=np.random.default_rng(17); diffs=[]
            idx=gr.index.intersection(gs.index)
            s2=gs.loc[idx,'e2'].to_numpy(); r2=gr.loc[idx,'e2'].to_numpy()
            for _ in range(REPS):
                p=rng.integers(0,len(idx),(len(idx),))
                diffs.append(np.sqrt(s2[p].mean())-np.sqrt(r2[p].mean()))
            rows.append(dict(subset=subset,arm=arm,guard=('tau50' if guard else 'none'),seeds=len(SEEDS),
                engines=int(len(idx)),rmse=float(np.sqrt(gs.e2.mean())),mae=float(gs.ae.mean()),
                nasa=float(gs.s.mean()),bias=float(gs.b.mean()),ref_rmse=float(np.sqrt(gr.e2.mean())),
                ref_mae=float(gr.ae.mean()),ref_nasa=float(gr.s.mean()),ref_bias=float(gr.b.mean()),
                delta=float(np.sqrt(gs.e2.mean())-np.sqrt(gr.e2.mean())),
                ci_lo=float(np.percentile(diffs,2.5)),ci_hi=float(np.percentile(diffs,97.5)),
                rel=float(100*(np.sqrt(gs.e2.mean())/np.sqrt(gr.e2.mean())-1))))
out=pd.DataFrame(rows)
pd.set_option('display.width',240)
print(out[['subset','arm','guard','engines','rmse','rel','mae','nasa','ref_mae','ref_nasa','ci_lo','ci_hi']].to_string(index=False,float_format=lambda x:'%.3f'%x))
out.to_csv(H/'SCHEDULE_FINAL.csv',index=False)