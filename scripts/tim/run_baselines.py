"""Standard uncertainty-quantification baselines that do not use the EAC construction.

The matrix in ``run_matrix.py`` compares conformal variants against the frozen
parametric layer.  A reviewer will also ask how the construction behaves against the
three things practitioners actually deploy:

  DE-Gauss   K-model deep ensemble, mean +- z * std, no calibration at all
  QR         quantile regression on the predicted RUL (piecewise-linear), no calibration
  CQR        conformalized quantile regression (Romano et al., 2019), cross-fitted

All three see exactly the same predictions as EAC: the same 15 trained models, the same
100 calibration engines, the same 93/235/97/228 test engines.

The quantile regressors are fitted by linear programming (exact, no solver licence):

    min_beta  sum_i rho_tau(y_i - x_i' beta)

which is the standard LP form with u_i, v_i >= 0 and x_i'beta + u_i - v_i = y_i.

Cross-fitting: the 100 calibration engines are split in two halves.  The quantile model
is fitted on one half and conformalized on the other, then the roles swap, so no engine
is used both to fit the quantile function and to set the conformal offset.  Three
different splits are averaged, matching the three-seed averaging used elsewhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
from run_matrix import scores_for_seed

LEVELS = (0.90, 0.50)
OUT = C.ROOT / "experiments" / "tim"


# ------------------------------------------------------------------ quantile regression
def qr_fit(X, y, tau):
    """Linear quantile regression by LP.  Returns beta with shape (p,)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    c = np.concatenate([np.zeros(p), np.full(n, tau), np.full(n, 1.0 - tau)])
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    res = optimize.linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"quantile regression failed: {res.message}")
    return res.x[:p]


def _design(centre, scale=None):
    """Piecewise-linear basis in the predicted RUL.

    A quadratic in the centre is badly conditioned (the columns are nearly collinear)
    and the LP then picks a degenerate vertex, which is why the first version of this
    script fitted quantile lines that were 30% too narrow.  A hinge basis with two knots
    is stable and flexible enough for a residual scale that changes by a factor of four
    across the support.  The scale column is dropped: it is a monotone function of the
    centre, so it adds collinearity without adding information.
    """
    u = np.asarray(centre, float) / C.MAX_RUL
    return np.column_stack([np.ones_like(u), u,
                            np.maximum(u - 1.0 / 3.0, 0.0),
                            np.maximum(u - 2.0 / 3.0, 0.0)])


def qr_interval(X_fit, y_fit, X_eval, level):
    a = 1.0 - level
    b_lo = qr_fit(X_fit, y_fit, a / 2.0)
    b_hi = qr_fit(X_fit, y_fit, 1.0 - a / 2.0)
    lo = np.clip(X_eval @ b_lo, 0, C.MAX_RUL)
    hi = np.clip(X_eval @ b_hi, 0, C.MAX_RUL)
    return lo, np.maximum(hi, lo)


def cqr_interval(X_fit, y_fit, X_cal, y_cal, X_eval, level):
    a = 1.0 - level
    lo_c, hi_c = qr_interval(X_fit, y_fit, X_cal, level)
    E = np.maximum(lo_c - y_cal, y_cal - hi_c)
    Q = C._q_upper(E, 1.0 - a)
    lo_e, hi_e = qr_interval(X_fit, y_fit, X_eval, level)
    lo = np.clip(lo_e - Q, 0, C.MAX_RUL)
    hi = np.clip(hi_e + Q, 0, C.MAX_RUL)
    return lo, np.maximum(hi, lo), float(Q)


# ------------------------------------------------------------------ aggregation helpers
def _metrics(lo, hi, y, lv):
    return dict(cov=C.coverage(lo, hi, y), width=C.width(lo, hi),
                winkler=C.winkler(lo, hi, y, lv))


def _agg(rows):
    out = {}
    for k in ("cov", "width", "winkler"):
        xs = [r[k] for r in rows]
        out[k] = float(np.mean(xs))
    for k in ("cov", "width"):
        xs = [r[k] for r in rows]
        out[f"{k}_sd"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return out


def run_subset(subset):
    dev, tst, layer = C.load_fleet(subset)
    y = layer["y"].astype(float)
    n = len(y)
    acc = {"DE-Gauss": {lv: [] for lv in LEVELS},
           "QR": {lv: [] for lv in LEVELS},
           "CQR": {lv: [] for lv in LEVELS}}

    # ---- deep ensemble over all 15 models, no calibration
    stack15 = np.stack([tst[(sd, f)][2] for sd in C.SEEDS for f in range(5)], axis=1)
    mu15 = stack15.mean(1)
    sd15 = stack15.std(1, ddof=1)
    for lv in LEVELS:
        z = stats.norm.ppf(0.5 + lv / 2.0)
        lo = np.clip(mu15 - z * sd15, 0, C.MAX_RUL)
        hi = np.clip(mu15 + z * sd15, 0, C.MAX_RUL)
        acc["DE-Gauss"][lv].append(_metrics(lo, hi, y, lv))

    # ---- QR / CQR, cross-fitted over two halves of the calibration engines
    for i, seed in enumerate(C.SEEDS):
        r_ens, p_ens, _ = scores_for_seed(dev, seed, ensemble=True)
        y_cal = p_ens + r_ens
        mu = np.stack([tst[(seed, f)][2] for f in range(5)], axis=1)
        pbar = mu.mean(1)
        Xc = _design(p_ens)
        Xe = _design(pbar)
        yt = tst[(seed, 0)][1]

        rng = np.random.default_rng(10_000 + 17 * i + len(y))
        perm = rng.permutation(len(y_cal))
        halves = (perm[: len(perm) // 2], perm[len(perm) // 2:])

        for lv in LEVELS:
            qr_m, cqr_m = [], []
            for h in (0, 1):
                fit, cal = halves[h], halves[1 - h]
                lo, hi = qr_interval(Xc[fit], y_cal[fit], Xe, lv)
                qr_m.append(_metrics(lo, hi, yt, lv))
                lo, hi, _ = cqr_interval(Xc[fit], y_cal[fit], Xc[cal], y_cal[cal], Xe, lv)
                cqr_m.append(_metrics(lo, hi, yt, lv))
            acc["QR"][lv].append({k: float(np.mean([m[k] for m in qr_m])) for k in qr_m[0]})
            acc["CQR"][lv].append({k: float(np.mean([m[k] for m in cqr_m])) for k in cqr_m[0]})

    rows = {}
    for name, per in acc.items():
        for lv, vals in per.items():
            agg = _agg(vals)
            agg["n"] = n
            rows[f"{name}@{lv:.2f}"] = agg
    return rows


def main():
    report = {"levels": list(LEVELS), "subsets": {}}
    for subset in C.SUBSETS:
        rows = run_subset(subset)
        report["subsets"][subset] = rows
        print(f"== {subset}")
        for k, v in rows.items():
            print(f"   {k:12s} cov {v['cov']:.3f}  w {v['width']:5.1f}  winkler {v['winkler']:6.1f}")
    dst = OUT / "baselines.json"
    dst.write_text(json.dumps(report, indent=1))
    print("wrote", dst)

    # fold the rows into matrix.json so that one file holds every construction
    mpath = OUT / "matrix.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        for subset, rows in report["subsets"].items():
            m["subsets"].setdefault(subset, {}).update(rows)
        mpath.write_text(json.dumps(m, indent=1))
        print("merged into", mpath)


if __name__ == "__main__":
    main()
