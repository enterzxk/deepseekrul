"""Raw-sequence test: PCA(20) + RBF kernel ridge, all fit inside CV folds."""
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
    X.append(w.reshape(-1)); y.append(T)
X = np.array(X, dtype=float); y = np.array(y, float)
mu, sd = X.mean(0), X.std(0)+1e-9
X = (X-mu)/sd
rng = np.random.default_rng(7); idx = rng.permutation(len(y)); folds = np.array_split(idx, 5)
def rbf(A, B, g):
    d2 = ((A[:,None,:]-B[None,:,:])**2).sum(-1)
    return np.exp(-g*d2)
best = (-9, None)
for ncomp in (10, 20, 40):
    for gamma in (0.002, 0.01):
        for lam in (0.3, 3.0):
            pred = np.zeros(len(y))
            ok = True
            for f in folds:
                tr = np.setdiff1d(idx, f)
                A = X[tr]; m = A.mean(0)
                U, S, Vt = np.linalg.svd(A-m, full_matrices=False)
                P = Vt[:ncomp].T
                Ztr = (A-m)@P; Zte = (X[f]-m)@P
                K = rbf(Ztr, Ztr, gamma)
                ymu = y[tr].mean()
                try:
                    a = np.linalg.solve(K + lam*np.eye(len(tr)), y[tr]-ymu)
                except np.linalg.LinAlgError:
                    ok = False; break
                pred[f] = rbf(Zte, Ztr, gamma) @ a + ymu
            if not ok: continue
            r2 = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
            if r2 > best[0]: best = (r2, (ncomp, gamma, lam))
print(f'best raw-sequence CV R^2 = {best[0]:+.3f} at (ncomp,gamma,lam)={best[1]}')
print('compare: summary-feature kernel R^2 = 0.000, ridge 0.036')
