"""Frozen metrics for LTAG v3 (Phase 0).

All estimators are explicit; never mix engine_mean and pooled in one column.
"""
import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float).reshape(-1)


def engine_mean_rmse(y, pred, engine_ids):
    y, pred, e = _arr(y), _arr(pred), np.asarray(engine_ids)
    vals = []
    for u in np.unique(e):
        m = e == u
        vals.append(np.sqrt(np.mean((pred[m] - y[m]) ** 2)))
    return float(np.mean(vals))


def pooled_rmse(y, pred):
    y, pred = _arr(y), _arr(pred)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def mae(y, pred):
    return float(np.mean(np.abs(_arr(pred) - _arr(y))))


def bias(y, pred):
    return float(np.mean(_arr(pred) - _arr(y)))


def nasa_sum(y, pred):
    d = _arr(pred) - _arr(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def nasa_mean(y, pred):
    return float(nasa_sum(y, pred) / len(_arr(y)))


def nasa_split(y, pred):
    d = _arr(pred) - _arr(y)
    late = np.where(d < 0, np.exp(-d / 13.0) - 1.0, 0.0)
    early = np.where(d >= 0, np.exp(d / 10.0) - 1.0, 0.0)
    return dict(late_sum=float(late.sum()), early_sum=float(early.sum()),
                late_mean=float(late.mean()), early_mean=float(early.mean()))


def per_engine_rmse(y, pred, engine_ids):
    y, pred, e = _arr(y), _arr(pred), np.asarray(engine_ids)
    out = {}
    for u in np.unique(e):
        m = e == u
        out[u] = float(np.sqrt(np.mean((pred[m] - y[m]) ** 2)))
    return out


def paired_engine_bootstrap(pred_a, pred_b, y, engine_ids, B=10000, seed=0):
    """Paired engine bootstrap of RMSE(A) - RMSE(B).

    Multi-seed callers must average predictions per engine across seeds before
    calling this function; resampling is over engines only.
    """
    y, a, b = _arr(y), _arr(pred_a), _arr(pred_b)
    e = np.asarray(engine_ids)
    units = np.unique(e)
    ea = np.array([np.sqrt(np.mean((a[e == u] - y[e == u]) ** 2)) for u in units])
    eb = np.array([np.sqrt(np.mean((b[e == u] - y[e == u]) ** 2)) for u in units])
    d = ea - eb
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    n = len(d)
    for i in range(B):
        boot[i] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_one = float(np.mean(boot >= 0.0))
    return dict(delta=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                p_one_sided=p_one, n_engines=int(n), B=int(B))


def sign_count(per_seed_delta):
    d = _arr(per_seed_delta)
    return dict(pos=int(np.sum(d > 0)), neg=int(np.sum(d < 0)), zero=int(np.sum(d == 0)),
                n=int(len(d)))
