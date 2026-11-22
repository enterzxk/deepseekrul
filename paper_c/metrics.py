"""Paper C metrics (single source of truth)."""
from __future__ import annotations

import numpy as np


def pooled_rmse(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def engine_mean_rmse(y, pred, engine):
    y, pred, e = np.asarray(y, float), np.asarray(pred, float), np.asarray(engine)
    vals = []
    for u in np.unique(e):
        m = e == u
        vals.append(np.sqrt(np.mean((pred[m] - y[m]) ** 2)))
    return float(np.mean(vals))


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(y, float))))


def nasa_score(y, pred):
    d = np.asarray(pred, float) - np.asarray(y, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0,
                                 np.exp(d / 10.0) - 1.0)))


def uncensored_slope_error(y, pred, engine, cut):
    """Mean absolute slope error on uncensored points, per engine.

    Uses the validation cut fractions as the time axis; only engines with at
    least three uncensored cuts are used.
    """
    y, pred, e, c = map(np.asarray, (y, pred, engine, cut))
    errs = []
    for u in np.unique(e):
        m = (e == u) & (y < 125.0)
        if m.sum() < 3:
            continue
        x = c[m].astype(float)
        if np.ptp(x) == 0:
            continue
        sp = np.polyfit(x, pred[m], 1)[0]
        st = np.polyfit(x, y[m], 1)[0]
        errs.append(abs(sp - st))
    return float(np.mean(errs)) if errs else float('nan')


def paired_engine_bootstrap(pred_a, pred_b, y, engine, B=10000, seed=0):
    y, a, b, e = map(np.asarray, (y, pred_a, pred_b, engine))
    units = np.unique(e)
    ea = np.array([np.sqrt(np.mean((a[e == u] - y[e == u]) ** 2)) for u in units])
    eb = np.array([np.sqrt(np.mean((b[e == u] - y[e == u]) ** 2)) for u in units])
    d = ea - eb
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return dict(delta=float(d.mean()),
                ci_lo=float(np.percentile(boot, 2.5)),
                ci_hi=float(np.percentile(boot, 97.5)),
                p_one_sided=float(np.mean(boot >= 0.0)))


def picp_mpiw(y, lo, hi):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return dict(picp=float(np.mean((y >= lo) & (y <= hi))),
                mpiw=float(np.mean(hi - lo)))
