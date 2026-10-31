"""Age-conditional survival prior for RUL (ECS design section 1.1).

Lifetime T is estimated on the training fleet of each subset (Weibull MLE, with a
Kaplan-Meier check). Given survival to age a, the RUL prior is the truncated
conditional distribution

    p0(R | a) = f_T(a + R) / S_T(a),   R >= 0

discretised on K bins over [0, cap]. The prior never uses a test engine's true
lifetime - only the fact that it has survived a cycles.
"""
from __future__ import annotations

import numpy as np


def fit_weibull(lifetimes: np.ndarray):
    """Method-of-moments start + Newton on the profile likelihood (no scipy needed)."""
    t = np.asarray(lifetimes, dtype=float)
    t = t[np.isfinite(t) & (t > 0)]
    n = t.size
    lt = np.log(t)
    mean_lt = lt.mean()
    # shape from the coefficient of variation (approximate, then 3 Newton steps)
    cv = t.std(ddof=1) / t.mean()
    k = max(0.5, min(10.0, 1.2 / max(cv, 1e-3)))
    for _ in range(60):
        num = (t ** k * lt).sum() / (t ** k).sum()
        f = 1.0 / k + mean_lt - num
        df = -1.0 / k ** 2 - ((t ** k * lt ** 2).sum() / (t ** k).sum() - num ** 2)
        step = f / df
        k_new = max(0.05, k - step)
        if abs(k_new - k) < 1e-8:
            k = k_new
            break
        k = k_new
    lam = ((t ** k).mean()) ** (1.0 / k)
    return float(k), float(lam)


def weibull_sf(t, k, lam):
    t = np.asarray(t, dtype=float)
    return np.exp(-(np.maximum(t, 0.0) / lam) ** k)


def weibull_pdf(t, k, lam):
    t = np.asarray(t, dtype=float)
    z = np.maximum(t, 1e-9) / lam
    return (k / lam) * z ** (k - 1) * np.exp(-(z ** k))


class AgeConditionalPrior:
    """Discretised p0(R | a) on a fixed grid, per subset."""

    def __init__(self, lifetimes, cap=125.0, n_bins=32):
        self.cap = float(cap)
        self.n_bins = int(n_bins)
        self.edges = np.linspace(0.0, cap, n_bins + 1)
        self.k, self.lam = fit_weibull(np.asarray(lifetimes, dtype=float))
        self.lifetimes = np.asarray(lifetimes, dtype=float)

    def prob(self, age: float) -> np.ndarray:
        """p0 over bins, shape (n_bins,), normalised."""
        a = max(float(age), 0.0)
        sa = float(weibull_sf(a, self.k, self.lam))
        sa = max(sa, 1e-9)
        lo = np.maximum(self.edges[:-1], 0.0)
        hi = self.edges[1:]
        # P(R in [lo, hi]) = [S(a+lo) - S(a+hi)] / S(a); last bin open-ended at cap
        hi_sf = weibull_sf(a + hi, self.k, self.lam)
        lo_sf = weibull_sf(a + lo, self.k, self.lam)
        p = (lo_sf - hi_sf) / sa
        p[-1] += float(weibull_sf(a + self.cap, self.k, self.lam) / sa)  # tail mass
        p = np.maximum(p, 1e-12)
        return p / p.sum()

    def median(self, age: float) -> float:
        p = self.prob(age)
        c = np.cumsum(p)
        idx = int(np.searchsorted(c, 0.5))
        idx = min(idx, self.n_bins - 1)
        return float(0.5 * (self.edges[idx] + self.edges[idx + 1]))

    def interval(self, age: float, lo=0.05, hi=0.95):
        p = self.prob(age)
        c = np.cumsum(p)
        i_lo = int(np.searchsorted(c, lo))
        i_hi = int(np.searchsorted(c, hi))
        return float(self.edges[min(i_lo, self.n_bins)]), float(
            self.edges[min(i_hi, self.n_bins)])


def km_sf(lifetimes: np.ndarray, grid: np.ndarray):
    """Kaplan-Meier survival check (all events observed -> step function of the ECDF)."""
    t = np.sort(np.asarray(lifetimes, dtype=float))
    n = t.size
    sf = 1.0 - np.searchsorted(t, grid, side="right") / n
    return np.clip(sf, 1.0 / n, 1.0)
