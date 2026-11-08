"""Why do two delta grids give different matched-coverage results?

Compares the sparse grid used in the paper (0.03..0.30) with a denser one
(0.02..0.40) on the same 80 cells, and reports the interpolation neighbours.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SPARSE = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
DENSE = [0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]
ROOT = Path("experiments/punet_matrix")


def split(u, frac=0.3, seed=42):
    rng = np.random.default_rng(seed); uu = np.unique(u).copy(); rng.shuffle(uu)
    n = max(1, int(round(uu.size * frac)))
    return np.isin(u, uu[:n]), np.isin(u, uu[n:])


def wink(y, lo, hi, a=0.1):
    return float(np.mean((hi - lo) + (2 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def q(v, lv):
    return float(np.quantile(v, min(1.0, lv), method="higher"))


def curve(y, lo, hi, cal, tst, grid):
    pts = []
    for dl in grid:
        E = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
        qq = q(E, (1 - dl) * (1 + 1 / max(cal.sum(), 1)))
        L, H = lo[tst] - qq, hi[tst] + qq
        pts.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), wink(y[tst], L, H)))
    return sorted(pts)


def curve_const(y, mu, cal, tst, grid):
    pts = []
    for dl in grid:
        s = np.abs(y[cal] - mu[cal])
        qq = q(s, (1 - dl) * (1 + 1 / max(cal.sum(), 1)))
        L, H = mu[tst] - qq, mu[tst] + qq
        pts.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), wink(y[tst], L, H)))
    return sorted(pts)


def at(pts, target=0.90):
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0), True, (c0, c1)
    best = min(pts, key=lambda p: abs(p[0] - target))
    return best[1], False, None


rows = []
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    for seed in [42, 7, 2024, 13]:
        for fold in range(5):
            d = ROOT / sub / f"f{fold}_s{seed}"
            if not (d / "preds_cqr.csv").exists() or not (d / "preds_point.csv").exists():
                continue
            c = np.genfromtxt(d / "preds_cqr.csv", delimiter=",", names=True)
            p = np.genfromtxt(d / "preds_point.csv", delimiter=",", names=True)
            cal, tst = split(c["unit"])
            y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
            mu = (p["y_lo"] + p["y_hi"]) / 2.0
            rec = dict(subset=sub, seed=seed, fold=fold)
            for tag, grid in (("sparse", SPARSE), ("dense", DENSE)):
                cq, ok1, br1 = at(curve(y, lo, hi, cal, tst, grid))
                ct, ok2, br2 = at(curve_const(p["y_true"], mu, cal, tst, grid))
                rec[f"cqr_{tag}"] = cq
                rec[f"const_{tag}"] = ct
                rec[f"ok_{tag}"] = ok1 and ok2
                rec[f"br_{tag}"] = br1
            rows.append(rec)

df = pd.DataFrame(rows)
print("cells:", len(df))
for tag in ("sparse", "dense"):
    rel = (df[f"cqr_{tag}"] - df[f"const_{tag}"]) / df[f"const_{tag}"] * 100
    print("%-7s wins %2d/%d  mean %+6.1f%%  median %+6.1f%%" % (
        tag, int((rel < 0).sum()), len(rel), rel.mean(), rel.median()))
print()
print("interpolation bracket for the dense grid (first 5):")
for _, r in df.head(5).iterrows():
    print("  %s s%d f%d: bracket %s" % (r.subset, r.seed, r.fold, r.br_dense))
df.to_csv("docs/tables/grid_sensitivity.csv", index=False)
