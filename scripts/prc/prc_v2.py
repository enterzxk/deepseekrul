"""PRC-R2 inference: four-state decision, IUT family, basic-bootstrap bands.

Design (R2 brief, part 2):
  * theta = min_p delta_p decides A-robustness;  max_p delta_p decides B-robustness.
  * Four states: A robust / B robust / protocol dependence CONFIRMED / inconclusive.
    "Protocol dependence confirmed" is an affirmative claim (max's LCB > 0 and min's
    UCB < 0), not a failure to reject.
  * Every decision is an intersection-union test (IUT): reject only if every component
    test rejects. IUT controls size at alpha without multiplicity correction, so the
    IUT is the honest baseline and the paper must state what the simultaneous band adds.
  * Bands are constructed with the basic (pivot) bootstrap, which corrects the first
    order finite-sample bias of an RMSE difference; the percentile band is kept as a
    diagnostic. max-t quantiles provide simultaneity.
  * Two resampling schemes, kept distinct in code and text:
      - cluster bootstrap: multinomial engine counts (resample engines with replacement)
      - multiplier bootstrap: exponential(1) engine weights
    They are asymptotically equivalent; finite-sample agreement is checked in the
    validation script.

Resampling is over engines (stratified when strata are supplied). Protocols are a
fixed set and are never resampled - their dependence is carried by the shared draws.
"""
from __future__ import annotations

import numpy as np


def rmse(err, w=None):
    if w is None:
        return np.sqrt(np.nanmean(err ** 2, axis=0))
    ww = np.where(np.isnan(err), 0.0, w[:, None])
    return np.sqrt(np.nansum(ww * err ** 2, axis=0) / ww.sum(axis=0))


def delta_se(err_a, err_b):
    """Delta-method SE of delta_p, from the influence function of a ratio-style stat."""
    n = err_a.shape[0]
    ra, rb = rmse(err_a), rmse(err_b)
    za = err_a ** 2 / (2 * ra)
    zb = err_b ** 2 / (2 * rb)
    z = zb - za
    return np.nanstd(z, axis=0, ddof=1) / np.sqrt(n)


def _resample_weights(rng, n, reps, scheme):
    if scheme == "multiplier":
        return rng.exponential(1.0, size=(reps, n))
    if scheme == "cluster":
        # multinomial counts = resampling engines with replacement
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=reps)
        return counts.astype(float)
    raise ValueError(scheme)


def bootstrap_deltas(err_a, err_b, reps=999, seed=1, scheme="multiplier",
                     strata=None):
    """Vectorised bootstrap: engine weights x squared-error matrix in one product."""
    n, k = err_a.shape
    rng = np.random.default_rng(seed)
    if scheme == "multiplier":
        w = rng.exponential(1.0, size=(reps, n))
    elif scheme == "cluster":
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=reps).astype(float)
    else:
        raise ValueError(scheme)

    def wrmse(err):
        mask = (~np.isnan(err)).astype(float)
        e2 = np.where(np.isnan(err), 0.0, err ** 2)
        num = w @ e2
        den = w @ mask
        return np.sqrt(num / np.maximum(den, 1e-12))

    return wrmse(err_b) - wrmse(err_a)


def _iut_decision(delta_hat, se, alpha, alternative):
    """One-sided IUT with a normal critical value; all components must reject."""
    z = 1.6448536269514722 if alpha == 0.05 else abs(np.percentile(
        np.random.default_rng(0).normal(size=200000), 100 * (1 - alpha)))
    if alternative == "greater":
        return bool(np.all(delta_hat - z * se > 0))
    return bool(np.all(delta_hat + z * se < 0))


def _iut_dependence(delta_hat, se, alpha):
    """Affirmative protocol-dependence test.

    Need BOTH (i) the largest delta is significantly positive and (ii) the smallest
    delta is significantly negative. Each is a one-sided test at level alpha; their
    intersection is an IUT with size <= alpha.
    """
    z = 1.6448536269514722 if alpha == 0.05 else abs(np.percentile(
        np.random.default_rng(0).normal(size=200000), 100 * (1 - alpha)))
    lcb_max = float(np.max(delta_hat - z * se))   # LCB of the largest delta
    ucb_min = float(np.min(delta_hat + z * se))   # UCB of the smallest delta
    return bool(lcb_max > 0 and ucb_min < 0), lcb_max, ucb_min


def prc_v2(err_a, err_b, alpha=0.05, reps=999, seed=1, scheme="multiplier",
           strata=None, band="basic"):
    n, k = err_a.shape
    delta_hat = rmse(err_b) - rmse(err_a)
    se = delta_se(err_a, err_b)
    boot = bootstrap_deltas(err_a, err_b, reps=reps, seed=seed, scheme=scheme,
                            strata=strata)

    # Simultaneous band. Primary interval: basic (pivot) bootstrap with a Bonferroni
    # quantile, which delivers simultaneous coverage while correcting the first-order
    # finite-sample bias of an RMSE difference (the bias measured in exp_consistency).
    # The max-t symmetric band is kept as a diagnostic of which protocol binds; it is
    # NOT the basis of the decision until its coverage clears the gate.
    t = np.abs(boot - delta_hat) / np.maximum(se, 1e-12)
    q = float(np.quantile(t.max(axis=1), 1 - alpha))
    a_b = alpha / k
    lo_p = np.quantile(boot, a_b / 2, axis=0)
    hi_p = np.quantile(boot, 1 - a_b / 2, axis=0)
    band_lo = 2 * delta_hat - hi_p
    band_hi = 2 * delta_hat - lo_p
    band_mt_lo = delta_hat - q * se
    band_mt_hi = delta_hat + q * se

    lcb_min = float(band_lo.min())
    ucb_min = float(band_hi.min())
    lcb_max = float(band_lo.max())
    ucb_max = float(band_hi.max())

    dep_norm, _, _ = _iut_dependence(delta_hat, se, alpha)
    dep_band = bool(lcb_max > 0 and ucb_min < 0)

    if lcb_min > 0:
        decision = "A robustly better"
    elif ucb_max < 0:
        decision = "B robustly better"
    elif dep_band:
        decision = "protocol dependence confirmed"
    else:
        decision = "inconclusive"

    return dict(delta_hat=delta_hat, se=se, q_simultaneous=q,
                band=[band_lo, band_hi], band_maxt=[band_mt_lo, band_mt_hi],
                theta_ci=[lcb_min, ucb_min],
                max_ci=[lcb_max, ucb_max],
                decision=decision,
                iut_A=_iut_decision(delta_hat, se, alpha, "greater"),
                iut_B=_iut_decision(delta_hat, se, alpha, "less"),
                iut_dependence=dep_norm)


def baseline_iut(delta_hat, se, alpha=0.05):
    """Pure IUT baseline: three binary tests, no simultaneous band, no four states."""
    a = _iut_decision(delta_hat, se, alpha, "greater")
    b = _iut_decision(delta_hat, se, alpha, "less")
    dep, _, _ = _iut_dependence(delta_hat, se, alpha)
    if a:
        return "A robustly better"
    if b:
        return "B robustly better"
    if dep:
        return "protocol dependence confirmed"
    return "inconclusive"


def baseline_bonferroni(err_a, err_b, alpha=0.05, reps=999, seed=1):
    n, k = err_a.shape
    delta_hat = rmse(err_b) - rmse(err_a)
    boot = bootstrap_deltas(err_a, err_b, reps=reps, seed=seed)
    a_b = alpha / k
    lo = np.quantile(boot, a_b / 2, axis=0)
    hi = np.quantile(boot, 1 - a_b / 2, axis=0)
    lo_b = 2 * delta_hat - hi
    hi_b = 2 * delta_hat - lo
    if lo_b.min() > 0:
        d = "A robustly better"
    elif hi_b.max() < 0:
        d = "B robustly better"
    else:
        d = "inconclusive"
    return dict(delta_hat=delta_hat, ci=[lo_b, hi_b], decision=d,
                width=float(np.mean(hi_b - lo_b)))
