"""Asymmetric conformal calibration: separate lower/upper quantiles.

Standard CQR adds one symmetric q_hat to both bounds. RUL errors are skewed
(the model over-predicts near end-of-life), so a symmetric interval wastes width
on one side. We calibrate each side separately:

    E_lo_i = q_lo_i - y_i      (how far the lower bound falls below the truth)
    E_hi_i = y_i - q_hi_i
    q_lo_hat = Quantile(E_lo, 1 - delta_lo)
    q_hi_hat = Quantile(E_hi, 1 - delta_hi),  delta_lo + delta_hi = delta

Symmetric CQR is the special case delta_lo = delta_hi = delta/2.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def split(u, frac=0.3, seed=42):
    rng = np.random.default_rng(seed); uu = np.unique(u).copy(); rng.shuffle(uu)
    n = max(1, int(round(uu.size * frac)))
    return np.isin(u, uu[:n]), np.isin(u, uu[n:])


def wink(y, lo, hi, a):
    return float(np.mean((hi - lo) + (2 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def q(v, lv):
    return float(np.quantile(v, min(1.0, lv), method="higher"))


def frontier(y, lo, hi, cal, tst, ratio):
    """ratio = share of delta spent on the lower side."""
    e_lo = lo[cal] - y[cal]
    e_hi = y[cal] - hi[cal]
    pts = []
    for dl in DELTAS:
        a_lo = dl * ratio
        a_hi = dl * (1 - ratio)
        ql = q(e_lo, 1 - a_lo)
        qh = q(e_hi, 1 - a_hi)
        L = lo[tst] - ql
        H = hi[tst] + qh
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
            rec = dict(subset=sub, seed=seed, fold=fold)
            for r in RATIOS:
                rec[f"r{r}"] = at(frontier(y, lo, hi, cal, tst, r))
            rows.append(rec)

df = pd.DataFrame(rows)
print("cells:", len(df))
base = "r0.5"
print("\n== mean Winkler by asymmetry ratio (lower share) ==")
for r in RATIOS:
    col = f"r{r}"
    rel = (df[col] - df[base]) / df[base] * 100
    wins = int((rel < 0).sum())
    print("  ratio %.1f: mean %6.1f | vs symmetric %+5.1f%% | better in %2d/%d" % (
        r, df[col].mean(), rel.mean(), wins, len(rel)))
best = min(RATIOS, key=lambda r: df[f"r{r}"].mean())
print("\nbest ratio: %.1f (mean %.1f vs symmetric %.1f)" % (best, df[f"r{best}"].mean(), df[base].mean()))
df.to_csv("docs/tables/asymmetric.csv", index=False)
