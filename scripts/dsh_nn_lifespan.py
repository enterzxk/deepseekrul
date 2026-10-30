"""Method-agnostic test: do similar mid-life windows have similar lifespans?"""
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
n = len(y)
d2 = ((Z[:,None,:]-Z[None,:,:])**2).sum(-1)
np.fill_diagonal(d2, np.inf)
nn = np.argsort(d2, axis=1)[:, :5]
sp = y[nn].std(axis=1)
print(f'n_engines={n}')
print(f'  lifespan SD overall              = {y.std():.1f}')
print(f'  median SD among 5 nearest windows = {np.median(sp):.1f}')
print(f'  ratio (NN SD / overall SD)        = {np.median(sp)/y.std():.2f}')
print()
print('If ratio ~1, similar windows give no lifespan information; if ~0, they do.')
