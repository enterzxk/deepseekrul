"""Seed ensembling of the quantile heads, with a matched ensembled baseline.

For each (subset, fold) the two seeds give two independent models. Averaging
their quantile outputs reduces the variance of the predicted width. The
constant-width baseline is ensembled the same way, so the comparison stays fair.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def split(u, frac=0.3, seed=42):
    rng = np.random.default_rng(seed); uu = np.unique(u).copy(); rng.shuffle(uu)
    n = max(1, int(round(uu.size * frac)))
    return np.isin(u, uu[:n]), np.isin(u, uu[n:])


def wink(y, lo, hi, a):
    return float(np.mean((hi - lo) + (2 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def q(v, lv):
    return float(np.quantile(v, min(1.0, lv), method="higher"))


def cqr_curve(y, lo, hi, cal, tst):
    pts = []
    for dl in DELTAS:
        E = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
        qq = q(E, (1 - dl) * (1 + 1 / E.size))
        L, H = lo[tst] - qq, hi[tst] + qq
        pts.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), wink(y[tst], L, H, dl)))
    return sorted(pts)


def scaled_curve(y, mu, w, cal, tst):
    pts = []
    for dl in DELTAS:
        s = np.abs(y[cal] - mu[cal]) / np.maximum(w[cal], 1e-9)
        qq = q(s, (1 - dl) * (1 + 1 / s.size))
        L, H = mu[tst] - qq * w[tst], mu[tst] + qq * w[tst]
        pts.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), wink(y[tst], L, H, dl)))
    return sorted(pts)


def at(pts, target=0.90):
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0)
    return min(pts, key=lambda p: abs(p[0] - target))[1]


root = Path("experiments/punet_matrix")
rows = []
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    for fold in range(5):
        ds = [root / sub / f"f{fold}_s{s}" for s in (42, 7)]
        if not all((d / "preds_cqr.csv").exists() and (d / "preds_point.csv").exists() for d in ds):
            continue
        qs = [np.genfromtxt(d / "preds_cqr.csv", delimiter=",", names=True) for d in ds]
        ps = [np.genfromtxt(d / "preds_point.csv", delimiter=",", names=True) for d in ds]
        # single seed 42
        c0, p0 = qs[0], ps[0]
        cal, tst = split(c0["unit"])
        y = c0["y_true"]
        single_cqr = at(cqr_curve(y, c0["y_lo"], c0["y_hi"], cal, tst))
        mu0 = (p0["y_lo"] + p0["y_hi"]) / 2.0
        single_const = at(scaled_curve(y, mu0, np.ones_like(mu0), cal, tst))
        # ensemble of two seeds
        lo = np.mean([x["y_lo"] for x in qs], axis=0)
        hi = np.mean([x["y_hi"] for x in qs], axis=0)
        mu = np.mean([(x["y_lo"] + x["y_hi"]) / 2.0 for x in ps], axis=0)
        ens_cqr = at(cqr_curve(y, lo, hi, cal, tst))
        ens_const = at(scaled_curve(y, mu, np.ones_like(mu), cal, tst))
        rows.append(dict(subset=sub, fold=fold, single_cqr=single_cqr, single_const=single_const,
                         ens_cqr=ens_cqr, ens_const=ens_const))

df = pd.DataFrame(rows)
print("folds:", len(df))
print(df.groupby("subset")[["single_cqr", "ens_cqr", "single_const", "ens_const"]].mean().round(1).to_string())
print()
for a, b, lab in [("single_cqr", "single_const", "single: CQR vs const"),
                  ("ens_cqr", "ens_const", "ensemble: CQR vs const"),
                  ("ens_cqr", "single_cqr", "ensemble CQR vs single CQR")]:
    rel = (df[a] - df[b]) / df[b] * 100
    print("  %-28s mean %+6.1f%% | better in %2d/%d" % (lab, rel.mean(), int((rel < 0).sum()), len(rel)))
df.to_csv("docs/tables/ensemble.csv", index=False)
