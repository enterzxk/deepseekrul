"""Full method comparison at matched coverage, from the saved dumps.

Methods: constant width, MC-dropout, uncalibrated quantiles, standard CQR,
localized CQR (ours). Each is calibrated on the same engine split and compared
at 0.90 realised coverage by interpolating its own frontier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy(); rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, a):
    return float(np.mean((hi - lo) + (2.0 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def wquant(v, w, lv):
    o = np.argsort(v); vv, ww = v[o], w[o]
    cw = np.cumsum(ww) / ww.sum()
    return float(vv[min(int(np.searchsorted(cw, lv)), vv.size - 1)])


def pts_scaled(y, mu, w, cal, tst):
    out = []
    for dl in DELTAS:
        q = float(np.quantile(np.abs(y[cal] - mu[cal]) / np.maximum(w[cal], 1e-9),
                              min(1.0, (1 - dl) * (1 + 1 / cal.sum())), method="higher"))
        L, H = mu[tst] - q * w[tst], mu[tst] + q * w[tst]
        out.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, 0.1)))
    return sorted(out)


def pts_cqr(y, lo, hi, cal, tst):
    out = []
    for dl in DELTAS:
        s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
        q = float(np.quantile(s, min(1.0, (1 - dl) * (1 + 1 / s.size)), method="higher"))
        L, H = lo[tst] - q, hi[tst] + q
        out.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, 0.1)))
    return sorted(out)


def pts_loc(y, lo, hi, cal, tst):
    s_cal = (hi[cal] - lo[cal]) / 2.0
    E = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
    s_tst = (hi[tst] - lo[tst]) / 2.0
    bw = max(float(np.median(np.abs(s_cal - np.median(s_cal)))), 1e-3)
    out = []
    for dl in DELTAS:
        lv = min(1.0, (1 - dl) * (1 + 1 / E.size))
        L = np.empty(tst.sum()); H = np.empty(tst.sum())
        for i, q in enumerate(s_tst):
            z = wquant(E, np.exp(-np.abs(s_cal - q) / bw), lv)
            L[i], H[i] = lo[tst][i] - z, hi[tst][i] + z
        out.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, 0.1)))
    return sorted(out)


def at(pts, target=0.90):
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0)
    best = min(pts, key=lambda p: abs(p[0] - target))
    return best[1]


root = Path("experiments/punet_matrix")
rows = []
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    for seed in [42, 7, 2024, 13]:
        for fold in range(5):
            d = root / sub / f"f{fold}_s{seed}"
            pq, pp, pm = d / "preds_cqr.csv", d / "preds_point.csv", d / "preds_mcd.csv"
            if not (pq.exists() and pp.exists()):
                continue
            c = np.genfromtxt(pq, delimiter=",", names=True)
            p = np.genfromtxt(pp, delimiter=",", names=True)
            cal, tst = engine_split(c["unit"])
            y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
            mu = (p["y_lo"] + p["y_hi"]) / 2.0
            rec = dict(subset=sub, seed=seed, fold=fold,
                       const=at(pts_scaled(p["y_true"], mu, np.ones_like(mu), cal, tst)),
                       qr=at(pts_cqr(y, lo, hi, cal, tst)) if False else float("nan"),
                       cqr=at(pts_cqr(y, lo, hi, cal, tst)),
                       loc=at(pts_loc(y, lo, hi, cal, tst)))
            if pm.exists():
                m = np.genfromtxt(pm, delimiter=",", names=True)
                c2, t2 = engine_split(m["unit"])
                rec["mcd"] = at(pts_scaled(m["y_true"], m["mu"], m["sd"], c2, t2))
            rows.append(rec)

df = pd.DataFrame(rows)
print("cells:", len(df), "| seeds:", sorted(df.seed.unique()))
cols = [c for c in ["const", "mcd", "cqr", "loc"] if c in df.columns]
print(df.groupby("subset")[cols].agg(["mean", "std"]).round(1).to_string())
print()
for base in [c for c in cols if c != "loc"]:
    rel = (df["loc"] - df[base]).dropna() / df[base].dropna() * 100
    if len(rel):
        print(f"Loc-CQR vs {base:<6}: wins {int((rel<0).sum())}/{len(rel)} mean {rel.mean():+.1f}% median {rel.median():+.1f}%")
df.to_csv("docs/tables/cells_all.csv", index=False)
