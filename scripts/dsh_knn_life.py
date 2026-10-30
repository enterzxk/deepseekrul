"""kNN regression with CV: is lifespan predictable after all?"""
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
rng = np.random.default_rng(7); idx = rng.permutation(len(y)); folds = np.array_split(idx, 5)
for k in (1, 3, 5, 10):
    pred = np.zeros(len(y))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        d2 = ((Z[f][:,None,:]-Z[tr][None,:,:])**2).sum(-1)
        nn = np.argsort(d2, axis=1)[:, :k]
        pred[f] = y[tr][nn].mean(1)
    ss_res = ((y-pred)**2).sum(); ss_tot = ((y-y.mean())**2).sum()
    print(f'  kNN k={k:2d}: CV R^2 = {1-ss_res/ss_tot:+.3f}')
print()
print('if any of these is well above 0.036, the ridge-based identifiability claim is too strong')
