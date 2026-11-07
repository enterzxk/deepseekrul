"""Evaluate PU-Net intervals end to end."""

from __future__ import annotations

import numpy as np

from .conformal import BlockConformalCalibrator, StratifiedCalibrator
from .diagnostics import engine_bootstrap_ci, stratified_coverage
from .metrics import engine_coverage, picp, pinaw, winkler


def evaluate_intervals(units, cutoffs, mu, y, y_hat, w, calib_mask, test_mask, delta: float = 0.1,
                       strata=None):
    """Calibrate on calib_mask, evaluate on test_mask."""
    units = np.asarray(units).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(y_hat, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)

    scores = BlockConformalCalibrator.scores(y[calib_mask], y_hat[calib_mask], w[calib_mask])
    if strata is not None:
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
