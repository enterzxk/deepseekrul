"""Non-linear (RBF kernel ridge) test: can lifespan be predicted from mid-life windows?"""
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
    X.append(np.concatenate([w.mean(0), w.std(0), w[-1]-w[0], w[-10:].mean(0)-w[:10].mean(0)])); y.append(T)
X = np.array(X); y = np.array(y, float)
Z = (X - X.mean(0)) / (X.std(0)+1e-9)
def rbf(A, B, g):
    d2 = ((A[:,None,:]-B[None,:,:])**2).sum(-1)
    return np.exp(-g*d2)
rng = np.random.default_rng(7); idx = rng.permutation(len(y)); folds = np.array_split(idx, 5)
best = (-9, None)
for gamma in (0.001, 0.005, 0.02):
    for lam in (0.1, 1.0, 10.0):
        pred = np.zeros(len(y))
        for f in folds:
            tr = np.setdiff1d(idx, f)
            K = rbf(Z[tr], Z[tr], gamma)
            mu = y[tr].mean()
            a = np.linalg.solve(K + lam*np.eye(len(tr)), y[tr]-mu)
            pred[f] = rbf(Z[f], Z[tr], gamma) @ a + mu
        r2 = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
        if r2 > best[0]: best = (r2, (gamma, lam))
        print(f'  gamma={gamma:6.3f} lam={lam:5.1f}: CV R^2 = {r2:+.3f}')
print()
print(f'best non-linear CV R^2 = {best[0]:+.3f} at {best[1]}')
print('compare: ridge 0.036, kNN -0.080')
