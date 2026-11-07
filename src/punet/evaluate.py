"""Evaluate PU-Net intervals end to end."""

from __future__ import annotations

import numpy as np

from .conformal import (BlockConformalCalibrator, StratifiedCalibrator, WeightedCalibrator, TwoComponentCalibrator)
from .diagnostics import engine_bootstrap_ci, stratified_coverage
from .metrics import engine_coverage, picp, pinaw, winkler


def evaluate_intervals(units, cutoffs, mu, y, y_hat, w, calib_mask, test_mask, delta: float = 0.1,
                       strata=None, weighted: bool = False, difficulty=None,
                       two_component: bool = False):
    """Calibrate on calib_mask, evaluate on test_mask."""
    units = np.asarray(units).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(y_hat, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)

    scores = BlockConformalCalibrator.scores(y[calib_mask], y_hat[calib_mask], w[calib_mask])
    if two_component:
        cal2 = TwoComponentCalibrator(delta=delta).fit(
            y[calib_mask], y_hat[calib_mask], w[calib_mask], units[calib_mask])
        lo, hi = cal2.interval(y_hat[test_mask], w[test_mask], units[test_mask])
        q_report = float(cal2.q_b)
        calib_mode = "two_component"
    elif weighted:
        if difficulty is None:
            raise ValueError("weighted calibration needs a difficulty covariate")
        difficulty = np.asarray(difficulty, dtype=np.float64).reshape(-1)
        cal = WeightedCalibrator(delta=delta).fit(scores, difficulty[calib_mask])
        lo, hi = cal.interval(y_hat[test_mask], w[test_mask], difficulty[test_mask])
        q_report = float(np.mean(np.abs(hi - lo) / (2 * w[test_mask])))
        calib_mode = "weighted"
    elif strata is not None:
        strata = np.asarray(strata).reshape(-1)
        cal = StratifiedCalibrator(delta=delta).fit(scores, strata[calib_mask])
        lo, hi = cal.interval(y_hat[test_mask], w[test_mask], strata[test_mask])
        q_report = float(np.mean(cal.quantiles(strata[test_mask])))
        calib_mode = "stratified"
    else:
        cal = BlockConformalCalibrator(delta=delta).fit(scores)
        lo, hi = cal.interval(y_hat[test_mask], w[test_mask])
        q_report = cal.q_hat
        calib_mode = "pooled"

    y_t = y[test_mask]
    covered = (y_t >= lo) & (y_t <= hi)
    cov = engine_coverage(units[test_mask], y_t, lo, hi)
    ci = engine_bootstrap_ci(units[test_mask], covered.astype(float))

    result = {
        "q_hat": q_report,
        "calib": calib_mode,
        "n_calib": int(np.sum(calib_mask)),
        "n_test": int(np.sum(test_mask)),
        "picp": picp(y_t, lo, hi),
        "pinaw": pinaw(y_t, lo, hi),
        "winkler": winkler(y_t, lo, hi, alpha=delta),
        "engine_min": cov["min"],
        "engine_p05": cov["p05"],
        "engine_median": cov["median"],
        "bootstrap_ci": list(ci),
        "rmse": float(np.sqrt(np.mean((y_t - y_hat[test_mask]) ** 2))),
    }
    result["diagnostics"] = stratified_coverage(
        units[test_mask], np.asarray(cutoffs).reshape(-1)[test_mask],
        np.asarray(mu, dtype=np.float64).reshape(-1)[test_mask], y_t, lo, hi,
    )
    return result


def evaluate_cqr_intervals(units, cutoffs, mu, y, q_lo, q_hi, calib_mask, test_mask, delta: float = 0.1):
    """Calibrate CQR on calib_mask and evaluate on test_mask."""
    from .cqr import CQRCalibrator

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    q_lo = np.asarray(q_lo, dtype=np.float64).reshape(-1)
    q_hi = np.asarray(q_hi, dtype=np.float64).reshape(-1)
    units = np.asarray(units).reshape(-1)

    scores = CQRCalibrator.scores(y[calib_mask], q_lo[calib_mask], q_hi[calib_mask])
    cal = CQRCalibrator(delta=delta).fit(scores)
    lo, hi = cal.interval(q_lo[test_mask], q_hi[test_mask])

    y_t = y[test_mask]
    covered = ((y_t >= lo) & (y_t <= hi)).astype(float)
    cov = engine_coverage(units[test_mask], y_t, lo, hi)
    ci = engine_bootstrap_ci(units[test_mask], covered)
    out = {
        "q_hat": cal.q_hat,
        "calib": "cqr",
        "n_calib": int(np.sum(calib_mask)),
        "n_test": int(np.sum(test_mask)),
        "picp": picp(y_t, lo, hi),
        "pinaw": pinaw(y_t, lo, hi),
        "winkler": winkler(y_t, lo, hi, alpha=delta),
        "engine_min": cov["min"],
        "engine_p05": cov["p05"],
        "engine_median": cov["median"],
        "bootstrap_ci": list(ci),
        "rmse": float(np.sqrt(np.mean((y_t - ((lo + hi) / 2)) ** 2))),
    }
    out["diagnostics"] = stratified_coverage(
        units[test_mask], np.asarray(cutoffs).reshape(-1)[test_mask],
        np.asarray(mu, dtype=np.float64).reshape(-1)[test_mask], y_t, lo, hi,
    )
    return out
