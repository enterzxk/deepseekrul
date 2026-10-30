"""A1: engine-clustered CI for C (resample (dataset,engine), keep all its cutoffs). No refitting."""
import numpy as np, pandas as pd, json
d = pd.read_csv('results/dsh/review_r2/C_predictions.csv')
print('rows:', len(d), '| engines:', d.groupby(['dataset','engine']).ngroups)
def rmse_of(frame, view, model):
    s = frame[(frame.view == view) & (frame.model == model)]
    e = s.y_pred - s.y_true
    return float(np.sqrt((e**2).mean()))
rng = np.random.default_rng(20260322)
out = {}
for model in ('ridge', 'rbf'):
    sub = d[d.model == model]
    keys = list(sub.groupby(['dataset','engine']).groups.keys())
    by_key = {k: g for k, g in sub.groupby(['dataset','engine'])}
    for scope in ('pooled', 'FD001', 'FD002'):
        kk = [k for k in keys if scope == 'pooled' or k[0] == scope]
        if len(kk) < 5: continue
        base_ = pd.concat([by_key[k] for k in kk])
        r1 = rmse_of(base_, 'v1', model); r2 = rmse_of(base_, 'v2', model)
        diffs = []
        for _ in range(2000):
            pick = [kk[i] for i in rng.integers(0, len(kk), len(kk))]
            b = pd.concat([by_key[k] for k in pick])
            diffs.append(rmse_of(b, 'v2', model) - rmse_of(b, 'v1', model))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[f'{model}|{scope}'] = dict(point=r2-r1, lo=float(lo), hi=float(hi), engines=len(kk), excludes0=bool(hi<0 or lo>0))
        print(f'{model:6s} {scope:7s} engines={len(kk):3d} dRMSE={r2-r1:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] excl0={hi<0 or lo>0}')
json.dump(out, open('results/dsh/review_r3/A1_C_clustered_ci.json','w'), indent=2)
