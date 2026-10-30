"""Stronger test: can a ridge model predict lifespan from the mid-life window?"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS
df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
cols = list(SELECTED_SENSOR_COLUMNS)
life = df.groupby('unit')['cycle'].max()
X, y = [], []
for u, g in df.groupby('unit'):
    g = g.sort_values('cycle'); T = int(life[u])
    end = max(50, int(round(T * 0.5)))
    w = g.iloc[end-50:end][cols].to_numpy(float)
    feats = np.concatenate([w.mean(0), w.std(0), w[-1]-w[0], w[-10:].mean(0)-w[:10].mean(0)])
    X.append(feats); y.append(T)
X = np.array(X); y = np.array(y, float)
mu, sd = X.mean(0), X.std(0)+1e-9
Z = (X-mu)/sd
rng = np.random.default_rng(7); idx = rng.permutation(len(y)); folds = np.array_split(idx, 5)
def ridge_fit(A, b, lam):
    return np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@b)
pred = np.zeros(len(y))
for f in folds:
    tr = np.setdiff1d(idx, f)
    w = ridge_fit(Z[tr], y[tr]-y[tr].mean(), 10.0)
    pred[f] = Z[f]@w + y[tr].mean()
ss_res = ((y-pred)**2).sum(); ss_tot = ((y-y.mean())**2).sum()
print(f'n_engines={len(y)}  features={X.shape[1]}')
print(f'  ridge CV R^2 for LIFESPAN  = {1-ss_res/ss_tot:+.3f}')
print(f'  baseline R^2 (predict mean) =  0.000')
print(f'  pred sd / true sd = {pred.std()/y.std():.3f}')
