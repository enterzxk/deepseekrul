"""Conformal calibration and evaluation for the two-level scale (numpy only).

The two-level model produces a per-window scale s_i = sigma_eng[e(i)] * sigma_win_i.
This module turns (centre, scale) into intervals and computes the metrics the pilot
pre-registers:

  * CQR-style score on the scaled residual:  s = max(lo0 - y, y - hi0) / scale
  * conformal quantile with the "higher" convention at level (1-delta)(1+1/n)
  * Winkler / PICP / per-engine coverage / oracle per-engine q (headroom ceiling)

With scale == 1 everywhere this reduces exactly to the frozen CQR procedure, which
the unit test checks.

Run: python scripts/test_twoscale_conformal.py
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, delta: float = 0.1) -> float:
    s = np.asarray(scores, float).reshape(-1)
    if s.size == 0:
        raise ValueError("empty calibration set")
    level = min(1.0, (1.0 - delta) * (1.0 + 1.0 / s.size))
    return float(np.quantile(s, level, method="higher"))


def cqr_scores(y, lo0, hi0, scale=None) -> np.ndarray:
    y = np.asarray(y, float).reshape(-1)
    lo0 = np.asarray(lo0, float).reshape(-1)
    hi0 = np.asarray(hi0, float).reshape(-1)
    s = np.maximum(lo0 - y, y - hi0)
    if scale is not None:
        sc = np.asarray(scale, float).reshape(-1)
        s = s / np.clip(sc, 1e-6, None)
    return s


def intervals(lo0, hi0, q, scale=None):
    """Expand the BASE interval [lo0, hi0] by q (optionally scaled per window).

    This is the CQR expansion: lo = lo0 - q, hi = hi0 + q.  With a per-window scale
    the expansion becomes lo = lo0 - q * scale, hi = hi0 + q * scale.
    """
    lo0 = np.asarray(lo0, float).reshape(-1)
    hi0 = np.asarray(hi0, float).reshape(-1)
    w = float(q) if scale is None else float(q) * np.asarray(scale, float).reshape(-1)
    return lo0 - w, hi0 + w


def winkler(y, lo, hi, alpha: float = 0.1) -> float:
    y = np.asarray(y, float).reshape(-1)
    lo = np.asarray(lo, float).reshape(-1)
    hi = np.asarray(hi, float).reshape(-1)
    return float(np.mean((hi - lo) + (2.0 / alpha) *
                         (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def picp(y, lo, hi) -> float:
    y = np.asarray(y, float).reshape(-1)
    return float(np.mean((y >= lo) & (y <= hi)))


def per_engine_coverage(y, lo, hi, units) -> dict:
    y = np.asarray(y, float).reshape(-1)
    units = np.asarray(units)
    cov = {}
    for u in np.unique(units):
        m = units == u
        cov[u] = float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
    v = np.array(list(cov.values()))
    return {"min": float(v.min()), "p05": float(np.quantile(v, 0.05)),
            "median": float(np.median(v)), "mean": float(v.mean())}


def oracle_engine_q(y, lo0, hi0, units, grid) -> dict:
    """Per-engine q that minimises that engine's Winkler (headroom ceiling)."""
    y = np.asarray(y, float).reshape(-1)
    lo0 = np.asarray(lo0, float).reshape(-1)
    hi0 = np.asarray(hi0, float).reshape(-1)
    units = np.asarray(units)
    out = {}
    for u in np.unique(units):
        m = units == u
        out[u] = float(min(grid, key=lambda q: winkler(y[m], lo0[m] - q, hi0[m] + q)))
    return out
