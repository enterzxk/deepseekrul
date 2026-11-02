"""Coverage auditing for interval-valued RUL predictions.

The unit of analysis is one engine at one cutoff: a pmf over the capped label grid, and
the observed label. Everything here works from (p, y) pairs, so it applies unchanged to
any interval method that can emit a pmf.

Two distinctions drive the whole audit and are kept explicit in the names:

  level error    m = alpha_hat - (1 - tau)          the marginal miss
  shape error    c_b = p_b - alpha_hat              the band-conditional residual

`m + c_b` is the total band miss. A test anchored on `alpha_hat` sees only `c`; a test
anchored on the nominal level sees `m + c`. Papers that report "band coverage agrees with
the nominal level" while the marginal is off by 0.07 are running the first test and
reading it as the second.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ------------------------------------------------------------------ intervals
def interval(p, level):
    """Shortest central equal-tailed interval on the label grid, from the pmf."""
    cdf = np.cumsum(p, axis=1)
    a = (1.0 - level) / 2.0
    n, K = p.shape
    lo = np.empty(n, int)
    hi = np.empty(n, int)
    for i in range(n):
        lo[i] = int(np.searchsorted(cdf[i], a))
        hi[i] = min(int(np.searchsorted(cdf[i], 1.0 - a)), K - 1)
    return lo, hi


def cover_indicator(p, y, level):
    lo, hi = interval(p, level)
    return (lo <= y) & (y <= hi), lo, hi


# ------------------------------------------------------------------ binomial basics
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def binom_p(k, n, p0):
    """Two-sided exact binomial p-value against p0."""
    if n == 0:
        return np.nan
    return float(stats.binomtest(int(k), int(n), p0).pvalue)


def mde(n, p0, alpha=0.05, power=0.80, two_sided=True):
    """Smallest coverage *deficit* a binomial test of size n detects with the given power.

    Only the downward direction is searched: the alternative of interest is "the band
    covers less than it claims", and an audit that reports only how easily it could
    detect over-coverage would answer a question nobody asked. The test itself remains
    two-sided, so the critical values use alpha/2 in each tail.

    Bisection rather than a grid because the critical values jump with n: a coarse scan
    makes delta* look ragged and non-monotone, which is an artifact of the search.

    Returns None when no attainable p1 reaches the target power -- i.e. when n is so small
    that even observing zero coverage cannot reject.
    """
    n = int(n)
    if n == 0:
        return None
    tail = alpha / (2 if two_sided else 1)
    c_lo = stats.binom.ppf(tail, n, p0) - 1
    c_hi = stats.binom.isf(tail, n, p0) + 1

    def pw(p1):
        return float(stats.binom.cdf(c_lo, n, p1) + (1 - stats.binom.cdf(c_hi, n, p1)))

    if pw(0.0) < power:
        return None
    lo, hi = 0.0, p0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pw(p0 - mid) >= power:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def power_at(n, p0, p1, alpha=0.05, two_sided=True):
    n = int(n)
    if n == 0:
        return np.nan
    crit_lo = stats.binom.ppf(alpha / (2 if two_sided else 1), n, p0) - 1
    crit_hi = stats.binom.isf(alpha / (2 if two_sided else 1), n, p0) + 1
    return float(stats.binom.cdf(crit_lo, n, p1) + (1 - stats.binom.cdf(crit_hi, n, p1)))


# ------------------------------------------------------------------ band table
def band_table(y, e, bands, anchor):
    """Per-band coverage with two reference points.

    anchor='marginal'  -> compare p_b against alpha_hat  (what the literature does)
    anchor='nominal'   -> compare p_b against the nominal level (the actual null)

    Both are returned so the caller can show they disagree.
    """
    a_hat = float(np.mean(e))
    rows = []
    for lo_b, hi_b in bands:
        m = (y >= lo_b) & (y <= hi_b) if hi_b >= y.max() else (y >= lo_b) & (y < hi_b)
        nb = int(m.sum())
        if nb == 0:
            continue
        kb = int(e[m].sum())
        pb = kb / nb
        wl, wh = wilson(kb, nb)
        ref = a_hat if anchor == "marginal" else None
        rows.append(dict(band=f"[{lo_b},{hi_b})", n=nb, k=kb, picp=pb,
                         wilson=[wl, wh], anchor=ref,
                         dev_vs_marginal=pb - a_hat,
                         mde_marginal=mde(nb, a_hat, alpha=0.05 / max(len(bands), 1)),
                         mde_nominal=mde(nb, 1 - 0.0, alpha=0.05 / max(len(bands), 1))))
    return rows


# ------------------------------------------------------------------ PIT
def randomized_pit(p, y, rng):
    """Randomised PIT for a discrete pmf.

    u = F(y-) + V * p(y),  V ~ U(0,1).

    Two things ride on this. For a discrete predictive distribution the non-randomised
    PIT F(y) is *not* uniform even when the model is perfect -- it is a step function
    evaluated at a random point. And for a capped label, p(last) is the surviving mass
    beyond the cap, so the same formula is exactly the right way to treat the atom: every
    capped engine is spread uniformly over the interval it is only known to lie in.
    """
    y = np.asarray(y, int)
    K = p.shape[1]
    cdf = np.cumsum(p, axis=1)
    lo = np.where(y > 0, cdf[np.arange(len(y)), np.clip(y - 1, 0, K - 1)], 0.0)
    mass = p[np.arange(len(y)), np.clip(y, 0, K - 1)]
    return lo + rng.random(len(y)) * mass


def naive_pit(p, y):
    """F(y) with no randomisation -- the version that piles capped engines on 1.0."""
    y = np.asarray(y, int)
    cdf = np.cumsum(p, axis=1)
    return cdf[np.arange(len(y)), np.clip(y, 0, p.shape[1] - 1)]


def pit_uniform(u):
    """Tests of u ~ U(0,1). KS is reported for familiarity; CvM is the one to trust."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    ks = stats.kstest(u, "uniform")
    cvm = stats.cramervonmises(u, "uniform")
    return dict(ks=float(ks.statistic), ks_p=float(ks.pvalue),
                cvm=float(cvm.statistic), cvm_p=float(cvm.pvalue),
                mean=float(u.mean()), var=float(u.var()),
                # share of the mass the non-randomised PIT parks on 1.0
                at_one=float(np.mean(u >= 1.0 - 1e-12)))


def pit_conditional(u, x, n_bins=5):
    """Is the PIT independent of the covariate?

    Three statistics, all using every observation and none of them a coverage count:

      spear   Spearman rank correlation between u and x. Detects monotone drift of the
              PIT with the prediction, which is what "over-confident at high RUL" looks
              like after the PIT transform.
      center  difference in mean |2u-1| between the top and bottom x-quintile. |2u-1| is
              sharpness: 0 means the label sat at the predictive median, 1 means it sat in
              a tail. A model that is too sharp at one end of the range shows up here and
              not in the coverage count, because the misses cancel.
      cvm_bin Cramer-von Mises of u against uniform, summed over x-bins.

    `spear` is the workhorse: one number, one test, no multiplicity, no binning of the
    outcome.
    """
    u = np.asarray(u, float)
    x = np.asarray(x, float)
    rho = stats.spearmanr(u, x)
    q = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    q[-1] += 1e-9
    cen = np.abs(2 * u - 1)
    idx = np.clip(np.searchsorted(q, x, side="right") - 1, 0, n_bins - 1)
    means = np.array([cen[idx == b].mean() if (idx == b).any() else np.nan
                      for b in range(n_bins)])
    cvm = 0.0
    for b in range(n_bins):
        if (idx == b).sum() >= 8:
            cvm += stats.cramervonmises(u[idx == b], "uniform").statistic
    return dict(spear=float(rho.statistic), spear_p=float(rho.pvalue),
                center_gap=float(np.nanmax(means) - np.nanmin(means)),
                center_by_bin=[float(v) for v in means],
                cvm_bin=float(cvm))


# ------------------------------------------------------------------ calibration slope
def calibration_slope(p, y, level=0.90):
    """Empirical coverage as a function of nominal level.

    A perfectly calibrated forecaster traces the diagonal. The gap between the curve and
    the diagonal, integrated, is a proper summary of miscalibration that does not depend
    on a chosen level -- and unlike a single-level coverage number it cannot be gamed by
    widening, because widening moves the whole curve down and to the right.
    """
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    emp = []
    for lv in levels:
        e, _, _ = cover_indicator(p, y, lv)
        emp.append(float(e.mean()))
    emp = np.array(emp)
    return levels, emp, float(np.mean(np.abs(emp - levels)))
