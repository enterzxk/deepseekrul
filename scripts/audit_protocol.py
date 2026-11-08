"""Audit the coverage-alignment protocol and the correlation statistics.

Two concerns:
  (A) The protocol interpolates each method's frontier to 0.90. Some cells never
      reach 0.90, and the interpolation may be extrapolation in disguise.
      Winkler is also averaged over WINDOWS, so long-lived engines dominate.
  (B) The width-error correlation is computed per window. Windows within an
      engine are not independent (pseudo-replication), which inflates any
      apparent association.

This script quantifies both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DELTAS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]
TARGET = 0.90
ROOT = Path("experiments/punet_matrix")


def split(u, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    uu = np.unique(u).copy(); rng.shuffle(uu)
    n = max(1, int(round(uu.size * frac)))
    return np.isin(u, uu[:n]), np.isin(u, uu[n:])


def wink_vec(y, lo, hi, a=0.1):
    return (hi - lo) + (2 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))


def q(v, lv):
    return float(np.quantile(v, min(1.0, lv), method="higher"))


def frontier(y, lo, hi, cal, tst, weight_by_engine=False, units=None):
    """Return (coverage, winkler) points; optionally engine-weighted Winkler."""
    pts = []
    for dl in DELTAS:
        E = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
        qq = q(E, (1 - dl) * (1 + 1 / max(cal.sum(), 1)))
        L, H = lo[tst] - qq, hi[tst] + qq
        cov = float(np.mean((y[tst] >= L) & (y[tst] <= H)))
        w = wink_vec(y[tst], L, H)
        if weight_by_engine and units is not None:
            ut = units[tst]
            per = [w[ut == u].mean() for u in np.unique(ut)]
            wl = float(np.mean(per))
        else:
            wl = float(w.mean())
        pts.append((cov, wl))
    return sorted(pts)


def at(pts, target=TARGET):
    """Interpolate; report whether the target was bracketed."""
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0), True
    best = min(pts, key=lambda p: abs(p[0] - target))
    return best[1], False


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
            units = c["unit"]
            mu = (p["y_lo"] + p["y_hi"]) / 2.0

            cqr_win, cqr_ok = at(frontier(y, lo, hi, cal, tst))
            cqr_eng, _ = at(frontier(y, lo, hi, cal, tst, True, units))
            const_win, const_ok = at(frontier(p["y_true"], mu - 0, mu + 0, cal, tst)) if False else (np.nan, False)
            # constant width: score = |y - mu|, interval mu +- q
            pts = []
            for dl in DELTAS:
                s = np.abs(y[cal] - mu[cal])
                qq = q(s, (1 - dl) * (1 + 1 / max(cal.sum(), 1)))
                L, H = mu[tst] - qq, mu[tst] + qq
                cov = float(np.mean((y[tst] >= L) & (y[tst] <= H)))
                w = wink_vec(y[tst], L, H)
                ut = units[tst]
                wl = float(np.mean([w[ut == u].mean() for u in np.unique(ut)]))
                pts.append((cov, float(w.mean()), wl))   # (cov, window, engine)
            pts.sort()
            const_win = at([(a, b) for a, b, _ in pts])[0]
            const_eng = at([(a, c_) for a, _, c_ in pts])[0]
            const_ok = at([(a, b) for a, b, _ in pts])[1]

            # correlation at window level vs engine level
            s_half = (hi - lo) / 2.0
            err = np.abs(y - (lo + hi) / 2.0)
            rho_win = float(np.corrcoef(s_half, err)[0, 1]) if s_half.std() > 1e-12 else np.nan
            engs = np.unique(units)
            sw = [s_half[units == u].mean() for u in engs]
            ew = [err[units == u].mean() for u in engs]
            rho_eng = float(np.corrcoef(sw, ew)[0, 1]) if np.std(sw) > 1e-12 and np.std(ew) > 1e-12 else np.nan

            rows.append(dict(subset=sub, seed=seed, fold=fold,
                             cqr_win=cqr_win, cqr_eng=cqr_eng,
                             const_win=const_win, const_eng=const_eng,
                             cqr_reached=cqr_ok, const_reached=const_ok,
                             rho_win=rho_win, rho_eng=rho_eng))

df = pd.DataFrame(rows)
print("cells:", len(df))
print("\n=== (A) coverage-alignment protocol ===")
print("cells where CQR frontier never reaches 0.90: %d/%d" % (int((~df.cqr_reached).sum()), len(df)))
print("cells where const frontier never reaches 0.90: %d/%d" % (int((~df.const_reached).sum()), len(df)))

for label, a, b in [("window-weighted", "cqr_win", "const_win"),
                    ("engine-weighted", "cqr_eng", "const_eng")]:
    rel = (df[a] - df[b]) / df[b] * 100
    print("  %-16s wins %2d/%d  mean %+6.1f%%  median %+6.1f%%" % (
        label, int((rel < 0).sum()), len(rel), rel.mean(), rel.median()))

print("\n=== (B) correlation statistics ===")
print("  window-level rho: mean %+.3f  median %+.3f  positive in %d/%d" % (
    df.rho_win.mean(), df.rho_win.median(), int((df.rho_win > 0).sum()), df.rho_win.notna().sum()))
print("  engine-level rho: mean %+.3f  median %+.3f  positive in %d/%d" % (
    df.rho_eng.mean(), df.rho_eng.median(), int((df.rho_eng > 0).sum()), df.rho_eng.notna().sum()))

df.to_csv("docs/tables/protocol_audit.csv", index=False)
