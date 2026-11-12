import numpy as np


def rmse(p, y):
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(p, y):
    return float(np.mean(np.abs(np.asarray(p, np.float64) - np.asarray(y, np.float64))))


def nasa_score(p, y):
    d = np.asarray(p, np.float64) - np.asarray(y, np.float64)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s)), float(np.mean(s))


def summarize(p, y):
    d = np.asarray(p, np.float64) - np.asarray(y, np.float64)
    nasa_sum, nasa_mean = nasa_score(p, y)
    return dict(rmse=rmse(p, y), mae=mae(p, y), nasa_sum=nasa_sum, nasa_mean=nasa_mean,
                bias=float(np.mean(d)), late_frac=float(np.mean(d > 0)),
                severe_late=float(np.mean(d > 10)), max_late=float(np.max(d)) if len(d) else 0.0)


def per_engine_rmse(unit, y_true, y_pred):
    """Engine-level RMSE, then equal weight per engine (audit R2).

    Three facts fix the relationship to the pooled ``rmse``:

    1. Under the frozen validation construction every engine contributes
       exactly five windows, so the pooled value already gives each engine the
       same weight ``1/E``. There is no row-count reweighting.
    2. The difference is the *order of aggregation*. Pooled is the root of the
       mean squared error, the engine-level value is the mean of the per-engine
       roots: ``sqrt(mean_e MSE_e) >= mean_e sqrt(MSE_e)`` by Jensen, so the
       pooled value is always >= the engine-level average.
    3. The gap comes from squaring, which amplifies the tail: a few engines
       with much larger squared errors move the pooled score more than they
       move the engine-level average.

    The task book makes the engine the statistical unit, so model selection
    uses this value. Added, not a replacement: ``rmse`` keeps its pooled
    meaning.
    """
    unit = np.asarray(unit)
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    if unit.size == 0:
        return float('nan')
    vals = [rmse(y_pred[unit == u], y_true[unit == u]) for u in np.unique(unit)]
    return float(np.mean(vals))


def per_engine_rmse_map(unit, y_true, y_pred):
    """Same quantity as ``per_engine_rmse`` but per engine, for diagnostics."""
    unit = np.asarray(unit)
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    out = {}
    for u in np.unique(unit):
        m = unit == u
        out[int(u)] = float(rmse(y_pred[m], y_true[m]))
    return out
