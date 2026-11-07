"""Interval quality metrics used in Part 3."""

from __future__ import annotations

import numpy as np


def _arrays(*xs):
    arrs = [np.asarray(x, dtype=np.float64).reshape(-1) for x in xs]
    if len({a.size for a in arrs}) != 1:
        raise ValueError("all inputs must have the same length")
    return arrs


def picp(y, lo, hi) -> float:
    """Prediction interval coverage probability (higher is better, target 1 - delta)."""
    y, lo, hi = _arrays(y, lo, hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def pinaw(y, lo, hi) -> float:
    """Normalised average interval width (lower is better)."""
    y, lo, hi = _arrays(y, lo, hi)
    span = float(np.max(y) - np.min(y))
    if span <= 0:
        raise ValueError("target range is zero; PINAW is undefined")
    return float(np.mean(hi - lo) / span)


def winkler(y, lo, hi, alpha: float = 0.1) -> float:
    """Winkler / interval score (lower is better)."""
    y, lo, hi = _arrays(y, lo, hi)
    width = hi - lo
    below = np.clip(lo - y, 0.0, None)
    above = np.clip(y - hi, 0.0, None)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def engine_coverage(units, y, lo, hi) -> dict:
    """Per-engine coverage summary: min / 5% / median / mean."""
    units, y, lo, hi = _arrays(units, y, lo, hi)
    covered = ((y >= lo) & (y <= hi)).astype(np.float64)
    rows = []
    for u in np.unique(units):
        mask = units == u
        rows.append(float(np.mean(covered[mask])))
    cov = np.asarray(rows)
    return {
        "n_engines": int(cov.size),
        "min": float(cov.min()),
        "p05": float(np.quantile(cov, 0.05)),
        "median": float(np.median(cov)),
        "mean": float(cov.mean()),
        "per_engine": cov,
    }
