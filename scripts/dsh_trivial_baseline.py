"""Do deep models beat a trivial linear model on the same windows?

If a ridge on a few hand-built features (residual norm, EMA, cycle) matches the
deep models on the identical Protocol-C windows, then the task is essentially
one-dimensional and architecture differences cannot matter much.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path("data/CMAPSS")
SENSORS = [f"s{i}" for i in range(1, 22)]
USE = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
OPS = ["op_1","op_2","op_3"]
CUTS = [60, 100, 140, 180, 220]

def build(subset="FD002"):
    df = pd.read_csv(ROOT / f"train_{subset}.txt", sep=r"\s+", header=None,
                     names=["unit","cycle"] + OPS + SENSORS)
    X, y, units = [], [], []
    for u, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        x = g[USE].to_numpy(float)
        early = x[: min(5, len(x))].mean(0)
        n = len(g)
        for c in CUTS:
            if n < c:
                continue
            w = x[max(0, c - 50):c]
            r = np.abs(w - early).mean()
            ema = w[-5:].mean(0) - early
            X.append([r, float(np.linalg.norm(ema)), c, r * c / 200.0])
            y.append(float(min(n - c, 125)))
            units.append(int(u))
    return np.array(X), np.array(y), np.array(units)

X, y, units = build()
rng = np.random.default_rng(7)
u = np.unique(units); perm = rng.permutation(u)
val_u = set(perm[: max(1, int(0.2 * len(u)))]); m = np.isin(units, list(val_u))
mu, sd = X[~m].mean(0), X[~m].std(0) + 1e-9
r = Ridge(alpha=1.0).fit((X[~m] - mu) / sd, y[~m])
p = r.predict((X[m] - mu) / sd)
print(f"ridge on 4 features | windows={m.sum()} engines={len(val_u)}")
print(f"  RMSE={np.sqrt(((p - y[m]) ** 2).mean()):.2f}  corr={np.corrcoef(p, y[m])[0,1]:.3f}")
print(f"  deep models (Protocol C pooled): 11.43-13.51, corr 0.95-0.96")
