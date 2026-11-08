"""Offline evaluation of every calibration variant from the saved dumps.

The training matrix stores, for each (subset, fold, seed), the raw quantile
outputs (preds_cqr.csv) and the point predictions (preds_point.csv). All
calibration variants are computed from these, so one training pass supports the
whole comparison table:

    PU-CQR          standard split CQR
    Loc-CQR         kernel-weighted CQR on the learned half-width
    QR (raw)        quantile heads without calibration
    constant-width  point prediction + global residual quantile
    MC-Dropout      point prediction + dropout std (computed separately)

Reported per cell: Winkler at the delta that yields ~0.90 realised coverage,
plus the realised coverage at that operating point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
TARGET = 0.90


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, alpha):
    return float(np.mean((hi - lo) + (2.0 / alpha) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def wquant(vals, w, level):
    o = np.argsort(vals)
    v, ww = vals[o], w[o]
    cw = np.cumsum(ww) / ww.sum()
    return float(v[min(int(np.searchsorted(cw, level)), v.size - 1)])


def cqr_pts(y, lo, hi, cal, tst):
    pts = []
    for dl in DELTAS:
        s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
        q = float(np.quantile(s, min(1.0, (1 - dl) * (1 + 1 / s.size)), method="higher"))
        L, H = lo[tst] - q, hi[tst] + q
        pts.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, dl)))
    return sorted(pts)


def loc_cqr_pts(y, lo, hi, cal, tst, bw_scale=1.0):
    s_cal = (hi[cal] - lo[cal]) / 2.0
    E = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
    s_tst = (hi[tst] - lo[tst]) / 2.0
    bw = max(float(np.median(np.abs(s_cal - np.median(s_cal)))) * bw_scale, 1e-3)
    pts = []
    for dl in DELTAS:
        lv = min(1.0, (1 - dl) * (1 + 1 / E.size))
        L = np.empty(tst.sum()); H = np.empty(tst.sum())
        for i, q in enumerate(s_tst):
            z = wquant(E, np.exp(-np.abs(s_cal - q) / bw), lv)
            L[i], H[i] = lo[tst][i] - z, hi[tst][i] + z
        pts.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, dl)))
    return sorted(pts)


def const_pts(y, mu, cal, tst):
    pts = []
    for dl in DELTAS:
        s = np.abs(y[cal] - mu[cal])
        q = float(np.quantile(s, min(1.0, (1 - dl) * (1 + 1 / s.size)), method="higher"))
        L, H = mu[tst] - q, mu[tst] + q
        pts.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, dl)))
    return sorted(pts)


def at(pts, target=TARGET):
    """Winkler interpolated at the target coverage.

    Returns (value, achieved_coverage, reached). When the frontier never reaches
    the target (a badly miscalibrated model), returns the closest operating
    point with reached=False instead of NaN, so the comparison stays defined.
    """
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0), target, True
    best = min(pts, key=lambda p: abs(p[0] - target))
    return best[1], best[0], False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/punet_matrix")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        for seed in args.seeds:
            for fold in range(args.folds):
                d = root / sub / f"f{fold}_s{seed}"
                pq, pp = d / "preds_cqr.csv", d / "preds_point.csv"
                if not (pq.exists() and pp.exists()):
                    continue
                c = np.genfromtxt(pq, delimiter=",", names=True)
                p = np.genfromtxt(pp, delimiter=",", names=True)
                cal, tst = engine_split(c["unit"])
                y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
                mu = p["y_true"] * 0 + (p["y_lo"] + p["y_hi"]) / 2.0
                c_v, c_cov, c_ok = at(cqr_pts(y, lo, hi, cal, tst))
                l_v, l_cov, l_ok = at(loc_cqr_pts(y, lo, hi, cal, tst))
                k_v, k_cov, k_ok = at(const_pts(p["y_true"], mu, cal, tst))
                rows.append(dict(
                    subset=sub, seed=seed, fold=fold,
                    cqr=c_v, loc=l_v, const=k_v,
                    cqr_cov=c_cov, loc_cov=l_cov, const_cov=k_cov,
                    all_reached=bool(c_ok and l_ok and k_ok),
                ))
    if not rows:
        print("no complete cells yet")
        return 0
    import pandas as pd
    df = pd.DataFrame(rows)
    print("cells:", len(df))
    print(df.groupby("subset")[["cqr", "loc", "const"]].agg(["mean", "std"]).round(1).to_string())
    print("\ncells where all methods reach 0.90:", int(df["all_reached"].sum()), "/", len(df))
    print()
    for col in ["cqr", "loc", "const"]:
        wins = (df["loc"] < df[col]).sum()
        print(f"loc-CQR better than {col}: {wins}/{len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
