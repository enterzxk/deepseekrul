"""Reconcile: compute CV residual SD directly vs within-neighbourhood SD."""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS
df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
cols = list(SELECTED_SENSOR_COLUMNS); life = df.groupby('unit')['cycle'].max()
X, y = [], []
for u, g in df.groupby('unit'):
    g = g.sort_values('cycle'); T = int(life[u])
    end = max(50, int(round(T*0.5)))
    w = g.iloc[end-50:end][cols].to_numpy(float)
    X.append(np.concatenate([w.mean(0), w.std(0), w[-1]-w[0]])); y.append(T)
X = np.array(X); y = np.array(y, float)
Z = (X - X.mean(0)) / (X.std(0)+1e-9)
d2 = ((Z[:,None,:]-Z[None,:,:])**2).sum(-1); np.fill_diagonal(d2, np.inf)
nn = np.argsort(d2, axis=1)[:, :5]
within = y[nn].std(axis=1, ddof=1)
print(f'mean within-neighbourhood SD (ddof=1) = {within.mean():.1f}   median = {np.median(within):.1f}')
print(f'overall lifespan SD                   = {y.std(ddof=1):.1f}')
print(f'-> implied R^2 from local consistency = {1-(np.median(within)/y.std(ddof=1))**2:.3f}')
print()
rng = np.random.default_rng(7); idx = rng.permutation(len(y)); folds = np.array_split(idx, 5)
pred = np.zeros(len(y))
for f in folds:
    tr = np.setdiff1d(idx, f)
    dd = ((Z[f][:,None,:]-Z[tr][None,:,:])**2).sum(-1)
    k = np.argsort(dd, axis=1)[:, :5]
    pred[f] = y[tr][k].mean(1)
resid = y - pred
print(f'CV residual SD                        = {resid.std(ddof=1):.1f}')
print(f'CV R^2                                = {1-(resid**2).sum()/((y-y.mean())**2).sum():+.3f}')
