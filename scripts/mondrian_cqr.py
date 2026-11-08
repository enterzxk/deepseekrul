"""Mondrian (stratified) conformal calibration by predicted-RUL bins.

A single global q_hat assumes the score distribution is the same everywhere.
If the error magnitude varies across the RUL range, calibrating one q_hat per
bin and applying it to test windows in the same bin should tighten intervals at
equal coverage.
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


def frontier(y, lo, hi, cal, tst, key, nbins):
    pts = []
    edges = np.quantile(key[cal], np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    cal_bin = np.clip(np.digitize(key[cal], edges) - 1, 0, nbins - 1)
    tst_bin = np.clip(np.digitize(key[tst], edges) - 1, 0, nbins - 1)
    E = np.maximum(lo - y, y - hi)
    for dl in DELTAS:
        L = lo[tst].copy(); H = hi[tst].copy()
        for b in range(nbins):
            m = cal_bin == b
            if m.sum() < 10:
                m = np.ones_like(cal_bin, dtype=bool)
            qq = q(E[cal][m], (1 - dl) * (1 + 1 / max(m.sum(), 1)))
            sel = tst_bin == b
            L[sel] = lo[tst][sel] - qq
            H[sel] = hi[tst][sel] + qq
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
    for seed in [42, 7]:
        for fold in range(5):
            d = root / sub / f"f{fold}_s{seed}"
            pq = d / "preds_cqr.csv"
            if not pq.exists():
                continue
            c = np.genfromtxt(pq, delimiter=",", names=True)
            cal, tst = split(c["unit"])
            y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
            rec = dict(subset=sub, seed=seed, fold=fold, global_=at(frontier(y, lo, hi, cal, tst, np.zeros_like(y), 1)))
            for nb in [3, 5, 10]:
                rec[f"mu{nb}"] = at(frontier(y, lo, hi, cal, tst, c["mu_T"], nb))
                rec[f"sg{nb}"] = at(frontier(y, lo, hi, cal, tst, c["sigma_T"], nb))
            rows.append(rec)

df = pd.DataFrame(rows)
print("cells:", len(df))
for col in ["mu3", "mu5", "mu10", "sg3", "sg5", "sg10"]:
    rel = (df[col] - df["global_"]) / df["global_"] * 100
    print("  %-6s mean %6.1f | vs global %+5.1f%% | better in %2d/%d" % (
        col, df[col].mean(), rel.mean(), int((rel < 0).sum()), len(rel)))
print("  global mean %.1f" % df["global_"].mean())
print()
print("per-subset best:")
for s in ["FD001", "FD002", "FD003", "FD004"]:
    d = df[df.subset == s]
    print("  %s global %.1f | mu5 %.1f | sg5 %.1f" % (s, d["global_"].mean(), d["mu5"].mean(), d["sg5"].mean()))
df.to_csv("docs/tables/mondrian.csv", index=False)
