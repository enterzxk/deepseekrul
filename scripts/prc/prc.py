"""PRC estimator: protocol-indexed simultaneous inference with three-state decision.

Implements paper/PRC_METHOD_DESIGN.md §3.

  delta_hat[p]      = RMSE_B(p) - RMSE_A(p)                 (positive = A better)
  theta_hat         = min_p delta_hat[p]
  band              = max-t simultaneous interval over protocols (cluster bootstrap)
  decision          = robust_A / robust_B / protocol_sensitive

All resampling is over engines (the exchangeable unit). Protocols are not resampled:
they are a fixed set, and their dependence is carried by the shared engine draws.
"""
from __future__ import annotations

import numpy as np


def rmse(err: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """err: (n_engine, k) -> per-protocol RMSE, optionally weighted."""
    if w is None:
        return np.sqrt((err ** 2).mean(axis=0))
    sw = w.sum()
    return np.sqrt((w[:, None] * err ** 2).sum(axis=0) / sw)


def _rng_weights(rng, n, reps):
    return rng.exponential(1.0, size=(reps, n))


def delta_se(err_a: np.ndarray, err_b: np.ndarray) -> np.ndarray:
    """Delta-method SE of delta_p = RMSE_B - RMSE_A (influence-function based).

    Used as the standardiser in the max-t band. A bootstrap-estimated SE in both the
    numerator and the standardiser makes the max-t quantile too small (self-normalised
    bootstrap is anti-conservative at larger n), which is what the coverage gate in
    PRC_METHOD_DESIGN section 5.3 caught.
    """
    n = err_a.shape[0]
    ra, rb = rmse(err_a), rmse(err_b)
    z = err_b ** 2 / (2 * rb) - err_a ** 2 / (2 * ra)
    return z.std(axis=0, ddof=1) / np.sqrt(n)


def prc_inference(err_a: np.ndarray, err_b: np.ndarray, alpha=0.05, reps=1999,
                  seed=20261101, strata: np.ndarray | None = None,
                  se_mode: str = "analytic"):
    """Protocol-robust comparison of two models.

    strata: optional integer array (e.g. subset id) — bootstrap resamples within strata.
    Returns delta_hat, theta interval (primary, direct bootstrap of min), the max-t
    band, and the three-state decision.
    """
    n, k = err_a.shape
    rng = np.random.default_rng(seed)
    delta_hat = rmse(err_b) - rmse(err_a)

    # bootstrap: resample engines (within strata if given), recompute all protocols
    boot = np.empty((reps, k))
    for b in range(reps):
        if strata is None:
            idx = rng.integers(0, n, n)
        else:
            idx = []
            for s in np.unique(strata):
                pool = np.flatnonzero(strata == s)
                idx.append(rng.choice(pool, pool.size, replace=True))
            idx = np.concatenate(idx)
        boot[b] = rmse(err_b[idx]) - rmse(err_a[idx])

    se = delta_se(err_a, err_b) if se_mode == "analytic" else boot.std(axis=0, ddof=1)
    # max-t simultaneous band: quantile of max_p |delta* - delta| / se
    t = np.abs(boot - delta_hat) / np.maximum(se, 1e-12)
    q = np.quantile(t.max(axis=1), 1 - alpha)
    band = np.column_stack([delta_hat - q * se, delta_hat + q * se])

    # theta = min_p delta_p. The percentile bootstrap of a minimum is negatively
    # biased (min is concave), so the band-based bounds are the primary inference:
    #   A robust  <=>  every protocol's LCB is above zero   (min_p LCB_p > 0)
    #   B robust  <=>  every protocol's UCB is below zero   (max_p UCB_p < 0)
    # The direct bootstrap of the minimum is reported as a diagnostic only.
    theta_hat = float(delta_hat.min())
    boot_min = boot.min(axis=1)
    theta_lo_direct = float(np.quantile(boot_min, alpha / 2))
    theta_hi_direct = float(np.quantile(boot_min, 1 - alpha / 2))
    theta_lo_band = float(band[:, 0].min())
    theta_hi_band = float(band[:, 1].max())
    theta_max_band = float(band[:, 1].max())   # UCB of max_p delta_p
    theta_max_lo = float(band[:, 0].max())     # LCB of max_p delta_p

    if theta_lo_band > 0:
        decision = "A robustly better"
    elif theta_max_band < 0:
        decision = "B robustly better"
    else:
        decision = "protocol-sensitive"

    support_a = [int(p) for p in np.flatnonzero(band[:, 0] > 0)]
    support_b = [int(p) for p in np.flatnonzero(band[:, 1] < 0)]

    return dict(delta_hat=delta_hat, se=se, band=band, q_simultaneous=float(q),
                theta_hat=theta_hat, theta_ci=[theta_lo_direct, theta_hi_direct],
                theta_ci_band=[theta_lo_band, theta_hi_band],
                max_delta_ci_band=[theta_max_lo, theta_max_band],
                decision=decision, protocols_supporting_A=support_a,
                protocols_supporting_B=support_b)


def naive_per_protocol(err_a, err_b, alpha=0.05, reps=1999, seed=20261101):
    """Baseline 1: independent percentile bootstrap per protocol + Bonferroni."""
    n, k = err_a.shape
    rng = np.random.default_rng(seed)
    delta_hat = rmse(err_b) - rmse(err_a)
    boot = np.empty((reps, k))
    for b in range(reps):
        idx = rng.integers(0, n, n)
        boot[b] = rmse(err_b[idx]) - rmse(err_a[idx])
    a_bonf = alpha / k
    lo = np.quantile(boot, a_bonf / 2, axis=0)
    hi = np.quantile(boot, 1 - a_bonf / 2, axis=0)
    if lo.min() > 0:
        decision = "A robustly better"
    elif hi.max() < 0:
        decision = "B robustly better"
    else:
        decision = "protocol-sensitive"
    theta_hat = float(delta_hat.min())
    return dict(delta_hat=delta_hat, ci=np.column_stack([lo, hi]),
                theta_hat=theta_hat,
                theta_ci=[float(lo.min()), float(hi.max())], decision=decision)


def moment_se(err_a, err_b):
    """Baseline 0 (plug-in): delta-method SE per protocol ignoring protocol dependence."""
    n = err_a.shape[0]
    ra, rb = rmse(err_a), rmse(err_b)
    za = err_a ** 2 / (2 * ra)          # influence of RMSE_A
    zb = err_b ** 2 / (2 * rb)
    z = zb - za
    se = z.std(axis=0, ddof=1) / np.sqrt(n)
    return se
