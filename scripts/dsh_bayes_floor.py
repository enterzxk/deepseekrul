"""Bayes-error estimate: how much headroom does the data allow at all?"""
import sys, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS
cols = list(SELECTED_SENSOR_COLUMNS)
df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
W = 50
X, y = [], []
for u, g in df.groupby('unit'):
    g = g.sort_values('cycle'); n = len(g)
    x = g[cols].to_numpy(float)
    for t in range(W, n + 1, 5):
        w = x[t-W:t]
        X.append(np.concatenate([w.mean(0), w.std(0), w[-1] - w[0]]))
        y.append(min(n - t, 125))
X = np.array(X); y = np.array(y, float)
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = (X - mu) / sd
print(f'windows={len(y)} dims={Z.shape[1]}')
rng = np.random.default_rng(7)
idx = rng.choice(len(y), size=min(3000, len(y)), replace=False)
Z = Z[idx]; y = y[idx]
# k-NN conditional variance in the informative range
for k in (5, 10, 20):
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    var = y[nn].var(axis=1)
    # Bayes RMSE ~ sqrt(mean conditional variance) over the sample
    print(f'k={k:2d}: mean cond var={var.mean():8.1f} -> implied floor RMSE={np.sqrt(var.mean()):5.2f}')
print(f'observed model RMSE ~ 12.9'  )
