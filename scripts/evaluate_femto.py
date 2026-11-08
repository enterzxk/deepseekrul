"""Evaluate the FEMTO cross-domain dumps with the same five methods.

Reads experiments/femto/<bearing>/preds_q.csv and preds_p.csv, calibrates on the
reserved calibration bearing, and reports each method at matched coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def winkler(y, lo, hi, a):
    return float(np.mean((hi - lo) + (2.0 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def wquant(v, w, lv):
    o = np.argsort(v); vv, ww = v[o], w[o]
    cw = np.cumsum(ww) / ww.sum()
    return float(vv[min(int(np.searchsorted(cw, lv)), vv.size - 1)])


def q_at(vals, dl):
    return float(np.quantile(vals, min(1.0, (1 - dl) * (1 + 1 / vals.size)), method="higher"))


def pts_cqr(y, lo, hi):
    out = []
    for dl in DELTAS:
        q = q_at(np.maximum(lo - y, y - hi), dl)
        L, H = lo - q, hi + q
        out.append((float(np.mean((y >= L) & (y <= H))), winkler(y, L, H, dl)))
    return sorted(out)


def pts_loc(y, lo, hi):
    s = (hi - lo) / 2.0
    E = np.maximum(lo - y, y - hi)
    bw = max(float(np.median(np.abs(s - np.median(s)))), 1e-3)
    out = []
    for dl in DELTAS:
        lv = min(1.0, (1 - dl) * (1 + 1 / E.size))
        L = np.empty(len(lo)); H = np.empty(len(lo))
        for i, q in enumerate(s):
            z = wquant(E, np.exp(-np.abs(s - q) / bw), lv)
            L[i], H[i] = lo[i] - z, hi[i] + z
        out.append((float(np.mean((y >= L) & (y <= H))), winkler(y, L, H, dl)))
    return sorted(out)


def pts_scaled(y, mu, w):
    out = []
    for dl in DELTAS:
        q = q_at(np.abs(y - mu) / np.maximum(w, 1e-9), dl)
        L, H = mu - q * w, mu + q * w
        out.append((float(np.mean((y >= L) & (y <= H))), winkler(y, L, H, dl)))
    return sorted(out)


def at(pts, target=0.90):
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0)
    return min(pts, key=lambda p: abs(p[0] - target))[1]


root = Path("experiments/femto")
rows = []
for d in sorted(root.glob("Bearing*")):
    pq, pp = d / "preds_q.csv", d / "preds_p.csv"
    if not (pq.exists() and pp.exists()):
        continue
    q = pd.read_csv(pq)
    p = pd.read_csv(pp)
    cal_q, tst_q = q["split"] == "cal", q["split"] == "test"
    cal_p, tst_p = p["split"] == "cal", p["split"] == "test"
    yq, lo, hi = q["y"].values, q["lo"].values, q["hi"].values
    yp, mu, sd = p["y"].values, p["mu"].values, p["sd"].values
    rows.append(dict(bearing=d.name,
                     cqr=at(pts_cqr(yq[tst_q], lo[tst_q], hi[tst_q])),
                     loc=at(pts_loc(yq[tst_q], lo[tst_q], hi[tst_q])),
                     const=at(pts_scaled(yp[tst_p], mu[tst_p], np.ones(tst_p.sum()))),
                     mcd=at(pts_scaled(yp[tst_p], mu[tst_p], sd[tst_p]))))

if not rows:
    print("no complete bearings yet")
    raise SystemExit(0)
df = pd.DataFrame(rows)
print(df.round(1).to_string(index=False))
print()
print(df[["cqr", "loc", "const", "mcd"]].agg(["mean", "std"]).round(1).to_string())
for base in ["const", "mcd", "cqr"]:
    rel = (df["loc"] - df[base]) / df[base] * 100
    print(f"Loc-CQR vs {base:<6}: wins {int((rel<0).sum())}/{len(df)} mean {rel.mean():+.1f}%")
df.to_csv("docs/tables/femto_cells.csv", index=False)
