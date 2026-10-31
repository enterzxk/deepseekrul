import glob, json, os
import numpy as np, pandas as pd
R = "/home/yndx/zxk/bearing-rul-pi/experiments"
SEEDS=[2027,2028,2029]; FOLDS=range(5)
ARMS={'A':('e10',True),'cnn1d':('e11',False),'bilstm':('e11',False),'gru':('e11',False)}
def load(arm,s,f):
    exp,f0=ARMS[arm]
    pat=(f"{R}/e3/{arm}_s{s}/FD002/{arm}_s{s}/matched_cutoff_predictions.csv" if f==0 and f0
         else f"{R}/{exp}/{arm}_s{s}_f{f}/FD002/{arm}_s{s}/matched_cutoff_predictions.csv")
    g=glob.glob(pat); return pd.read_csv(g[0]).sort_values('unit_id').reset_index(drop=True) if g else None
data={}
for arm in ARMS:
    per_fold=[]
    ok=True
    for f in FOLDS:
        stack=[]
        for s in SEEDS:
            d=load(arm,s,f)
            if d is None: ok=False; break
            stack.append(d.signed_error.to_numpy(float))
        if not ok: break
        per_fold.append(np.stack(stack,axis=1))   # (52 engines, 3 seeds)
    if ok: data[arm]=per_fold
print("complete arms:", list(data))
def pool(arm): return np.concatenate(data[arm],axis=0)   # (260, 3)
for a in data: print("%-8s pooled RMSE %.3f" % (a, np.sqrt(np.mean(pool(a)**2))))
# bootstrap A vs each
REPS=20000; rng=np.random.default_rng(9000)
sizes=[52]*5
res={}
for b in [x for x in data if x!='A']:
    ea=pool('A'); eb=pool(b)
    pt=np.sqrt(np.mean(eb**2))-np.sqrt(np.mean(ea**2))
    acc=np.empty(REPS)
    for r in range(REPS):
        idx=np.concatenate([f*52+rng.integers(0,52,52) for f in range(5)])
        acc[r]=np.sqrt(np.mean(eb[idx]**2))-np.sqrt(np.mean(ea[idx]**2))
    lo,hi=np.percentile(acc,[2.5,97.5])
    res[b]=dict(delta=float(pt),ci=[float(lo),float(hi)],excludes_zero=bool(lo>0 or hi<0))
    print("%-8s - A = %+.3f  CI [%+.3f,%+.3f]  excludes0=%s" % (b,pt,lo,hi,res[b]['excludes_zero']))
os.makedirs(f"{R}/../results/dsh/e15",exist_ok=True)
json.dump(dict(arms=list(data),contrasts=res), open(f"{R}/../results/dsh/e15/arch_contrasts_partial.json","w"), indent=2)
print("wrote (partial: tcn/tf pending)")
