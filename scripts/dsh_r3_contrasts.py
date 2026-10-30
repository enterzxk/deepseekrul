"""Fast engine-clustered contrasts from saved R3 predictions."""
import numpy as np, pandas as pd, json
d = pd.read_csv('results/dsh/review_r3/R3_preds.csv')
d['se'] = (d.y_pred - d.y_true)**2
piv = {}
def npair(a, b, scope):
    sub = d if scope == 'pooled' else d[d.dataset == scope]
    A = sub[sub.arm == a].set_index(['dataset','engine','cutoff']).se
    B = sub[sub.arm == b].set_index(['dataset','engine','cutoff']).se
    idx = A.index.intersection(B.index)
    A = A.loc[idx]; B = B.loc[idx]
    eng = pd.Series([i[0]+'|'+str(i[1]) for i in idx], index=idx)
    return A, B, eng
def clustered(a, b, scope, reps=2000, seed=7):
    A, B, eng = npair(a, b, scope)
    codes, uniq = pd.factorize(eng.to_numpy())
    n = len(uniq)
    sumsA = np.bincount(codes, weights=A.to_numpy(), minlength=n)
    sumsB = np.bincount(codes, weights=B.to_numpy(), minlength=n)
    cnt = np.bincount(codes, minlength=n).astype(float)
    rng = np.random.default_rng(seed)
    base_a = np.sqrt(A.mean()); base_b = np.sqrt(B.mean())
    diffs = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, n, n)
        ca = cnt[pick].sum(); diffs[i] = np.sqrt(sumsB[pick].sum()/ca) - np.sqrt(sumsA[pick].sum()/ca)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(point=float(base_b - base_a), lo=float(lo), hi=float(hi), engines=int(n), excl0=bool(hi < 0 or lo > 0))
pairs = {'cycle_minus_nocycle': ('arm_cycle','arm_nocycle'), 'pe_minus_nocycle': ('arm_pe','arm_nocycle'),
         'v2_minus_v1': ('v2','v1'), 'v1_minus_v0': ('v1','v0')}
out = {}
for name, (a, b) in pairs.items():
    for scope in ('pooled','FD002'):
        o = clustered(a, b, scope); out[f'{name}|{scope}'] = o
        print(f'{name:22s} {scope:7s} engines={o["engines"]:3d} dRMSE={o["point"]:+.4f} CI[{o["lo"]:+.4f},{o["hi"]:+.4f}] excl0={o["excl0"]}')
json.dump(out, open('results/dsh/review_r3/R3_contrasts.json','w'), indent=2)
