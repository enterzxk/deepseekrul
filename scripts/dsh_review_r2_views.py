"""R2-C: V0/V1/V2 views, ridge + RBF, engine-level 5-fold, exactly 30 outer fits."""
import sys, warnings, json, itertools
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS

CUTS = (60, 100, 140); WIN = 50; CAP = 125.0
cols = list(SELECTED_SENSOR_COLUMNS)
frames = {ds: load_cmapss_train(Path('data/CMAPSS'), ds) for ds in ('FD001','FD002')}

def build(ds, df):
    life = df.groupby('unit')['cycle'].max()
    settings = [c for c in df.columns if c.startswith('setting')][:3]
    rows = []
    for u, g in df.groupby('unit'):
        T = int(life[u]); g = g.sort_values('cycle').reset_index(drop=True)
        for c in CUTS:
            if T <= c or c < WIN: continue
            w = g.iloc[c-WIN:c]
            if len(w) < WIN: continue
            sens = w[cols].to_numpy(float)
            v0 = sens.reshape(-1)
            last = g.iloc[c-1]
            cyc = float(c)
            setv = last[settings].to_numpy(float) if len(settings) else np.zeros(3)
            # observable cluster id from settings (nearest of 6 fixed centres is not available offline; use raw settings)
            v1 = np.concatenate([v0, [cyc], setv])
            # prefix summaries up to t=c only
            def slope(a):
                t = np.arange(len(a), dtype=float); return np.polyfit(t, a, 1)[0]
            s50 = np.array([slope(sens[:, j]) for j in range(len(cols))])
            hist = g.iloc[max(0, c-100):c][cols].to_numpy(float)
            s100 = np.array([slope(hist[:, j]) for j in range(len(cols))]) if len(hist) >= 20 else s50.copy()
            dstd = np.array([np.diff(hist[:, j]).std() if len(hist) > 2 else 0.0 for j in range(len(cols))])
            v2 = np.concatenate([v1, s50, s100, dstd])
            rows.append(dict(dataset=ds, engine=int(u), cutoff=c, life=T,
                             rul=min(float(T - c), CAP), v0=v0, v1=v1, v2=v2))
    return rows

rows = []
for ds, df in frames.items(): rows += build(ds, df)
print(f'observations: {len(rows)}  engines: {len(set((r["dataset"], r["engine"]) for r in rows))}')
from collections import Counter
print('by cutoff:', dict(Counter(r['cutoff'] for r in rows)))

def ridge_fit(A, b, lam):
    return np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@b)
def rbf(A, B, g):
    d2 = ((A[:,None,:]-B[None,:,:])**2).sum(-1)
    return np.exp(-g*d2)

MODELLED = {'ridge': None, 'rbf': None}
results = []; nfit = 0
eng_keys = sorted(set((r['dataset'], r['engine']) for r in rows))
rng = np.random.default_rng(2026); perm = rng.permutation(len(eng_keys)); folds = np.array_split(perm, 5)
fold_of = {}
for k, f in enumerate(folds):
    for i in f: fold_of[eng_keys[i]] = k
for r in rows: r['fold'] = fold_of[(r['dataset'], r['engine'])]

for view in ('v0','v1','v2'):
    for model in ('ridge','rbf'):
        for k in range(5):
            tr = [r for r in rows if r['fold'] != k]; te = [r for r in rows if r['fold'] == k]
            Xtr = np.array([r[view] for r in tr], float); Xte = np.array([r[view] for r in te], float)
            ytr = np.array([r['rul'] for r in tr], float)
            mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1.0
            A, B = (Xtr-mu)/sd, (Xte-mu)/sd
            if model == 'ridge':
                w = ridge_fit(A, ytr-ytr.mean(), 10.0); pred = B@w + ytr.mean()
            else:
                dm = ((A[:,None,:]-A[None,:,:])**2).sum(-1)
                g = 1.0/ (np.median(dm[dm>0]) + 1e-12)
                K = rbf(A, A, g); ym = ytr.mean()
                a = np.linalg.solve(K + 1.0*np.eye(len(tr)), ytr-ym)
                pred = rbf(B, A, g) @ a + ym
            nfit += 1
            for r, p in zip(te, pred):
                results.append(dict(dataset=r['dataset'], engine=r['engine'], cutoff=r['cutoff'],
                                    view=view, model=model, fold=k, y_true=r['rul'], y_pred=float(p)))
print('outer fits executed:', nfit)
assert nfit == 30, nfit
out = pd.DataFrame(results)
out.to_csv('results/dsh/review_r2/C_predictions.csv', index=False)
print(out.groupby(['view','model']).size().to_string())
