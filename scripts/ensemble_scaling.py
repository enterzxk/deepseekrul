"""Seed-ensemble scaling: how the gain grows with the number of models.

For each (subset, fold) we ensemble the quantile heads of N seeds and compare
the resulting CQR interval against a constant-width baseline ensembled the same
way. Reported at matched coverage of 0.90.
"""

import itertools
import numpy as np
import pandas as pd
from pathlib import Path

DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
SEEDS = [42, 7, 2024, 13]


def split(u, frac=0.3, seed=42):
    rng = np.random.default_rng(seed); uu = np.unique(u).copy(); rng.shuffle(uu)
    n = max(1, int(round(uu.size * frac)))
    return np.isin(u, uu[:n]), np.isin(u, uu[n:])


def wink(y, lo, hi, a):
    return float(np.mean((hi - lo) + (2 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def q(v, lv):
    return float(np.quantile(v, min(1.0, lv), method="higher"))


def cqr_pts(y, lo, hi, cal, tst):
    pts = []
    for dl in DELTAS:
        E = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
        qq = q(E, (1 - dl) * (1 + 1 / E.size))
        L, H = lo[tst] - qq, hi[tst] + qq
        pts.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), wink(y[tst], L, H, dl)))
    return sorted(pts)


def const_pts(y, mu, cal, tst):
    pts = []
    for dl in DELTAS:
        s = np.abs(y[cal] - mu[cal])
        qq = q(s, (1 - dl) * (1 + 1 / s.size))
        L, H = mu[tst] - qq, mu[tst] + qq
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
        avail = [s for s in SEEDS
                 if (root / sub / f"f{fold}_s{s}" / "preds_cqr.csv").exists()
                 and (root / sub / f"f{fold}_s{s}" / "preds_point.csv").exists()]
        if len(avail) < 2:
            continue
        qs = {s: np.genfromtxt(root / sub / f"f{fold}_s{s}" / "preds_cqr.csv", delimiter=",", names=True) for s in avail}
        ps = {s: np.genfromtxt(root / sub / f"f{fold}_s{s}" / "preds_point.csv", delimiter=",", names=True) for s in avail}
        ref = qs[avail[0]]
        cal, tst = split(ref["unit"])
        y = ref["y_true"]
        for n in (1, 2, len(avail)):
            if n > len(avail):
                continue
            use = avail[:n]
            lo = np.mean([qs[s]["y_lo"] for s in use], axis=0)
            hi = np.mean([qs[s]["y_hi"] for s in use], axis=0)
            mu = np.mean([(ps[s]["y_lo"] + ps[s]["y_hi"]) / 2.0 for s in use], axis=0)
            rows.append(dict(subset=sub, fold=fold, n=n,
                             cqr=at(cqr_pts(y, lo, hi, cal, tst)),
                             const=at(const_pts(y, mu, cal, tst))))

df = pd.DataFrame(rows)
print("folds per n:", df.groupby("n").size().to_dict())
print()
print("n   CQR     const   rel      wins")
for n in sorted(df.n.unique()):
    d = df[df.n == n]
    rel = (d["cqr"] - d["const"]) / d["const"] * 100
    print("%-3d %6.1f  %6.1f  %+6.1f%%  %2d/%d" % (n, d["cqr"].mean(), d["const"].mean(), rel.mean(), int((rel < 0).sum()), len(rel)))
print()
print("per-subset at n=%d:" % df.n.max())
d = df[df.n == df.n.max()]
for s in ["FD001", "FD002", "FD003", "FD004"]:
    e = d[d.subset == s]
    rel = (e["cqr"] - e["const"]) / e["const"] * 100
    print("  %s cqr %.1f const %.1f rel %+.1f%%" % (s, e["cqr"].mean(), e["const"].mean(), rel.mean()))
df.to_csv("docs/tables/ensemble_scaling.csv", index=False)
