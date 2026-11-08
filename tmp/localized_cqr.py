"""Localized CQR: kernel-weighted conformal calibration on the CQR score.

Idea: the quantile heads already predict a per-sample half-width s_i, and the
localized-CP baseline shows that local calibration helps. Combining them uses
s_i as the similarity key: calibration scores are weighted by how close their
predicted half-width is to the query's, then a weighted quantile is taken.

This keeps the CQR interval shape and adds local adaptivity without any
retrieval structure.
"""
import numpy as np
from pathlib import Path

DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy(); rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, alpha):
    return float(np.mean((hi - lo) + (2.0 / alpha) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def wquant(vals, weights, level):
    o = np.argsort(vals); v, w = vals[o], weights[o]
    cw = np.cumsum(w) / w.sum()
    return float(v[min(int(np.searchsorted(cw, level)), v.size - 1)])


def frontier(y, lo, hi, cal, tst, bw_scale=1.0):
    s_cal = (hi[cal] - lo[cal]) / 2.0
    E_cal = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
    s_tst = (hi[tst] - lo[tst]) / 2.0
    bw = max(float(np.median(np.abs(s_cal - np.median(s_cal)))) * bw_scale, 1e-3)
    pts = []
    for dl in DELTAS:
        lv = min(1.0, (1 - dl) * (1 + 1 / E_cal.size))
        L = np.empty(tst.sum()); H = np.empty(tst.sum())
        for i, q in enumerate(s_tst):
            w = np.exp(-np.abs(s_cal - q) / bw)
            z = wquant(E_cal, w, lv)
            L[i], H[i] = lo[tst][i] - z, hi[tst][i] + z
        pts.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, dl)))
    return sorted(pts)


def at(pts, target=0.90):
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0)
    return float("nan")


root = Path("experiments/punet_e2e")
print("%-6s %-14s %-10s" % ("subset", "method", "W@cov0.90"))
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    d = root / sub / "f0_s42" / "preds.csv"
    if not d.exists():
        print(sub, "missing"); continue
    c = np.genfromtxt(d, delimiter=",", names=True)
    cal, tst = engine_split(c["unit"])
    y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
    print("%-6s %-14s %-10.1f" % (sub, "Loc-CQR", at(frontier(y, lo, hi, cal, tst))))
    print("%-6s %-14s %-10.1f" % (sub, "Loc-CQR bw2", at(frontier(y, lo, hi, cal, tst, 2.0))))
