"""R2-E: does drift measured BEFORE the window improve current RUL? Two specs, 10 fits."""
import sys, warnings, json
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
            sig_a = za.mean(1); sig_b = zb.mean(1)
            mu_prior = float(np.polyfit(np.arange(W, dtype=float), sig_a, 1)[0])
            sd_prior = float(np.diff(sig_a).std())
            t = np.arange(W, dtype=float)
            feats_b = np.concatenate([zb.mean(0), zb.std(0), zb[-1], [np.polyfit(t, zb[:, j], 1)[0] for j in range(len(cols))]])
            rows.append(dict(dataset=ds, engine=int(u), cutoff=c, cycle=float(c),
                             s1=feats_b, mu_prior=mu_prior, sd_prior=sd_prior,
                             rul=min(float(T-c), CAP)))
d = rows
print(f'observations={len(d)} engines={len({(r["dataset"], r["engine"]) for r in d})}')
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

preds = {}; nfit = 0
for spec in ('S1', 'S2'):
    for k in range(5):
        tr = [r for r in d if r['fold'] != k]; te = [r for r in d if r['fold'] == k]
        def fx(r):
            base = np.concatenate([r['s1'], [r['cycle']]])
            return np.concatenate([base, [r['mu_prior'], r['sd_prior']]]) if spec == 'S2' else base
        Xtr = np.array([fx(r) for r in tr]); Xte = np.array([fx(r) for r in te])
        ytr = np.array([r['rul'] for r in tr])
        p = fit(Xtr, ytr, Xte); nfit += 1
        for r, v in zip(te, p):
            preds[(spec, r['dataset'], r['engine'], r['cutoff'])] = (float(v), r['rul'], spec)
print('outer fits:', nfit); assert nfit == 10

def rmse(spec, sel):
    v = [preds[(spec, a, b, c)][0] - preds[(spec, a, b, c)][1] for (a, b, c) in sel]
    return float(np.sqrt(np.mean(np.square(v))))
allk = sorted({(a, b, c) for (s, a, b, c) in preds})
r1, r2 = rmse('S1', allk), rmse('S2', allk)
print(f'RMSE S1={r1:.3f}  S2={r2:.3f}  diff={r2-r1:+.3f}')
boot = []
for _ in range(2000):
    pick = [allk[i] for i in rng.integers(0, len(allk), len(allk))]
    boot.append(rmse('S2', pick) - rmse('S1', pick))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f'paired V2-V1: {r2-r1:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  excludes0={bool(hi<0 or lo>0)}')
json.dump({'rmse_s1': r1, 'rmse_s2': r2, 'point': r2-r1, 'lo': float(lo), 'hi': float(hi),
           'n_obs': len(allk), 'excludes0': bool(hi<0 or lo>0)},
          open('results/dsh/review_r2/E_prior_history.json','w'), indent=2)
