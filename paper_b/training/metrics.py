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
