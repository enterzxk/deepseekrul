"""A2: reproduce E exactly (2 specs x 5 folds, unchanged), save per-object predictions, then engine-clustered CI."""
import sys, warnings, json, hashlib
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS

CUTS = (100, 140); W = 50; CAP = 125.0
cols = list(SELECTED_SENSOR_COLUMNS)
rows = []
for ds in ('FD001', 'FD002'):
    df = load_cmapss_train(Path('data/CMAPSS'), ds)
    life = df.groupby('unit')['cycle'].max()
    for u, g in df.groupby('unit'):
        T = int(life[u]); g = g.sort_values('cycle').reset_index(drop=True)
        early = g.iloc[:20][cols].to_numpy(float)
        m0, s0 = early.mean(0), early.std(0); s0[s0 < 1e-9] = 1.0
        for c in CUTS:
            if T <= c or c < 2*W: continue
            A = g.iloc[c-2*W:c-W][cols].to_numpy(float)
            B = g.iloc[c-W:c][cols].to_numpy(float)
            za = (A-m0)/s0; zb = (B-m0)/s0
            sig_a = za.mean(1)
            mu_prior = float(np.polyfit(np.arange(W, dtype=float), sig_a, 1)[0])
            sd_prior = float(np.diff(sig_a).std())
            t = np.arange(W, dtype=float)
            feats_b = np.concatenate([zb.mean(0), zb.std(0), zb[-1], [np.polyfit(t, zb[:, j], 1)[0] for j in range(len(cols))]])
            rows.append(dict(dataset=ds, engine=int(u), cutoff=c, cycle=float(c), s1=feats_b,
                             mu_prior=mu_prior, sd_prior=sd_prior, rul=min(float(T-c), CAP)))
d = rows
keys = sorted({(r['dataset'], r['engine']) for r in d})
rng = np.random.default_rng(2026); perm = rng.permutation(len(keys)); folds = np.array_split(perm, 5)
fold_of = {}
for k, f in enumerate(folds):
    for i in f: fold_of[keys[i]] = k
for r in d: r['fold'] = fold_of[(r['dataset'], r['engine'])]
def fit(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1.0
    A, B = (Xtr-mu)/sd, (Xte-mu)/sd
    w = np.linalg.solve(A.T@A + 10.0*np.eye(A.shape[1]), A.T@(ytr-ytr.mean()))
    return B@w + ytr.mean()
recs = []; nfit = 0
for spec in ('S1', 'S2'):
    for k in range(5):
        tr = [r for r in d if r['fold'] != k]; te = [r for r in d if r['fold'] == k]
        def fx(r):
            b_ = np.concatenate([r['s1'], [r['cycle']]])
            return np.concatenate([b_, [r['mu_prior'], r['sd_prior']]]) if spec == 'S2' else b_
        Xtr = np.array([fx(r) for r in tr]); Xte = np.array([fx(r) for r in te])
        ytr = np.array([r['rul'] for r in tr])
        p = fit(Xtr, ytr, Xte); nfit += 1
        for r, v in zip(te, p):
            recs.append(dict(spec=spec, dataset=r['dataset'], engine=r['engine'], cutoff=r['cutoff'],
                             fold=k, y_true=r['rul'], y_pred=float(v)))
print('fits:', nfit); assert nfit == 10
out = pd.DataFrame(recs); out.to_csv('results/dsh/review_r3/E_predictions.csv', index=False)
def rmse(df, spec):
    s = df[df.spec == spec]; e = s.y_pred - s.y_true
    return float(np.sqrt((e**2).mean()))
print(f'pooled RMSE S1={rmse(out,"S1"):.4f} S2={rmse(out,"S2"):.4f} diff={rmse(out,"S2")-rmse(out,"S1"):+.4f}')
rng2 = np.random.default_rng(77); res = {}
for scope in ('pooled','FD001','FD002'):
    sub = out if scope == 'pooled' else out[out.dataset == scope]
    kk = list(sub.groupby(['dataset','engine']).groups.keys())
    by = {k: g for k, g in sub.groupby(['dataset','engine'])}
    diffs = []
    for _ in range(2000):
        pick = [kk[i] for i in rng2.integers(0, len(kk), len(kk))]
        b_ = pd.concat([by[k] for k in pick])
        diffs.append(rmse(b_, 'S2') - rmse(b_, 'S1'))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    res[scope] = dict(point=rmse(sub,'S2')-rmse(sub,'S1'), lo=float(lo), hi=float(hi), engines=len(kk), excludes0=bool(hi<0 or lo>0))
    print(f'{scope:7s} engines={len(kk):3d} dRMSE={res[scope]["point"]:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] excl0={res[scope]["excludes0"]}')
json.dump(res, open('results/dsh/review_r3/A2_E_clustered_ci.json','w'), indent=2)
print('script sha256:', hashlib.sha256(open(__file__,'rb').read()).hexdigest()[:16])
