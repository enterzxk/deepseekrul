"""Block split conformal calibration for temporally dependent RUL data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockSplit:
    fit: np.ndarray
    tune: np.ndarray
    calib: np.ndarray
    test: np.ndarray

    def describe(self) -> dict:
        return {
            "fit": int(self.fit.size),
            "tune": int(self.tune.size),
            "calib": int(self.calib.size),
            "test": int(self.test.size),
        }


def block_split(
    units: np.ndarray,
    cutoffs: np.ndarray,
    *,
    tune_frac: float = 0.15,
    calib_frac: float = 0.2,
    seed: int = 42,
    fit_units_frac: float = 0.6,
) -> BlockSplit:
    """Split by engine first, then by adjacent time blocks inside each group.

    Engines are assigned to fit / tune / calib groups so that no engine leaks
    across groups. Within the calibration group, samples are ordered by cutoff
    so the calibration block stays temporally adjacent to the test region.
    """
    units = np.asarray(units)
    cutoffs = np.asarray(cutoffs)
    if units.shape != cutoffs.shape:
        raise ValueError("units and cutoffs must share a shape")

    rng = np.random.default_rng(seed)
    uniq = np.unique(units)
    rng.shuffle(uniq)

    n = uniq.size
    n_fit = max(1, int(round(n * fit_units_frac)))
    n_tune = max(1, int(round(n * tune_frac)))
    if n_fit + n_tune >= n:
        raise ValueError("too few engines for the requested split")
    fit_u = uniq[:n_fit]
    tune_u = uniq[n_fit : n_fit + n_tune]
    calib_u = uniq[n_fit + n_tune :]

    fit_idx = np.flatnonzero(np.isin(units, fit_u))
    tune_idx = np.flatnonzero(np.isin(units, tune_u))
    calib_idx = np.flatnonzero(np.isin(units, calib_u))

    # adjacent time blocks inside the calibration group
    order = calib_idx[np.argsort(cutoffs[calib_idx])]
    n_cal = max(1, int(round(order.size * calib_frac)))
    calib_part = order[-n_cal:]
    test_part = order[:-n_cal]
    if test_part.size == 0:
        raise ValueError("calibration fraction consumed the whole group")

    split = BlockSplit(fit=fit_idx, tune=tune_idx, calib=calib_part, test=test_part)
    _assert_disjoint(split)
    return split


def _assert_disjoint(split: BlockSplit) -> None:
    sets = [set(split.fit.tolist()), set(split.tune.tolist()), set(split.calib.tolist()), set(split.test.tolist())]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                raise AssertionError(f"split blocks {i} and {j} overlap")


class BlockConformalCalibrator:
    """Split conformal on scale-normalised scores."""

    def __init__(self, delta: float = 0.1) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        self.delta = float(delta)
        self._q = None
        self._n = 0

    @staticmethod
    def scores(y_true: np.ndarray, y_hat: np.ndarray, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)
        if np.any(w <= 0):
            raise ValueError("scale must be strictly positive")
        return np.abs(np.asarray(y_true) - np.asarray(y_hat)) / w

    def fit(self, scores: np.ndarray) -> "BlockConformalCalibrator":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size == 0:
            raise ValueError("empty calibration set")
        self._n = int(scores.size)
        level = min(1.0, (1.0 - self.delta) * (1.0 + 1.0 / self._n))
        self._q = float(np.quantile(scores, level, method="higher"))
        return self

    @property
    def q_hat(self) -> float:
        if self._q is None:
            raise RuntimeError("calibrator is not fitted")
        return self._q

    def interval(self, y_hat: np.ndarray, w: np.ndarray):
        y_hat = np.asarray(y_hat, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        half = self.q_hat * w
        return y_hat - half, y_hat + half


def engine_calib_split(units: np.ndarray, *, calib_frac: float = 0.3, seed: int = 42):
    """Engine-disjoint calibration / test split.

    Whole engines are assigned to either the calibration pool or the test pool,
    so the calibration set spans the full lifetime (early and late windows)
    instead of a single time block. This is the split used by the main
    experiments; block_split is kept for the time-block ablation.
    """
    units = np.asarray(units)
    rng = np.random.default_rng(seed)
    uniq = np.unique(units)
    rng.shuffle(uniq)
    n_cal = max(1, int(round(uniq.size * calib_frac)))
    if n_cal >= uniq.size:
        raise ValueError("calib_frac leaves no test engines")
    calib_units = uniq[:n_cal]
    test_units = uniq[n_cal:]
    calib = np.flatnonzero(np.isin(units, calib_units))
    test = np.flatnonzero(np.isin(units, test_units))
    if set(calib.tolist()) & set(test.tolist()):
        raise AssertionError("engine split leaked")
    return calib, test


class StratifiedCalibrator:
    """Mondrian (stratified) split conformal.

    Engine-level heterogeneity makes marginal exchangeability fail: a pool of
    calibration engines can be systematically harder than the test engines, so
    one global quantile over-covers. Fitting one quantile per stratum restores
    coverage inside each stratum and removes that systematic bias.

    Args:
        delta: nominal miscoverage.
        min_count: strata with fewer calibration points fall back to the pooled
            quantile, so a tiny stratum cannot produce a degenerate interval.
    """

    def __init__(self, delta: float = 0.1, min_count: int = 30) -> None:
        self.delta = float(delta)
        self.min_count = int(min_count)
        self._q: dict = {}
        self._fallback = None

    def fit(self, scores: np.ndarray, strata: np.ndarray) -> "StratifiedCalibrator":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        strata = np.asarray(strata).reshape(-1)
        if scores.size != strata.size:
            raise ValueError("scores and strata must share a length")

        def q_of(vals: np.ndarray) -> float:
            n = vals.size
            level = min(1.0, (1.0 - self.delta) * (1.0 + 1.0 / n))
            return float(np.quantile(vals, level, method="higher"))

        self._fallback = q_of(scores)
        self._q = {}
        for s in np.unique(strata):
            vals = scores[strata == s]
            if vals.size >= self.min_count:
                self._q[s] = q_of(vals)
        return self

    def quantiles(self, strata: np.ndarray) -> np.ndarray:
        strata = np.asarray(strata).reshape(-1)
        if self._fallback is None:
            raise RuntimeError("calibrator is not fitted")
        return np.array([self._q.get(s, self._fallback) for s in strata], dtype=np.float64)

    def interval(self, y_hat: np.ndarray, w: np.ndarray, strata: np.ndarray):
        y_hat = np.asarray(y_hat, dtype=np.float64).reshape(-1)
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        half = self.quantiles(strata) * w
        return y_hat - half, y_hat + half


class WeightedCalibrator:
    """Locally weighted split conformal (Barber et al., 2023).

    Engine populations differ in difficulty, so a pool of calibration engines
    can be easier than the test engines and one global quantile under-covers.
    Weighting each calibration score by its similarity to the query point
    replaces marginal exchangeability with a bounded coverage gap under a
    weighted total-variation condition.

    The weight kernel uses a difficulty covariate available at prediction time
    (here: the predicted RUL, i.e. the life stage). Bandwidth defaults to the
    median absolute deviation of the covariate.
    """

    def __init__(self, delta: float = 0.1, bandwidth: float | None = None) -> None:
        self.delta = float(delta)
        self.bandwidth = bandwidth
        self._scores = None
        self._difficulty = None

    def fit(self, scores: np.ndarray, difficulty: np.ndarray) -> "WeightedCalibrator":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        difficulty = np.asarray(difficulty, dtype=np.float64).reshape(-1)
        if scores.size != difficulty.size:
            raise ValueError("scores and difficulty must share a length")
        order = np.argsort(scores)
        self._scores = scores[order]
        self._difficulty = difficulty[order]
        if self.bandwidth is None:
            mad = float(np.median(np.abs(difficulty - np.median(difficulty))))
            self.bandwidth = max(mad, 1e-3)
        return self

    def _weighted_quantile(self, query: float) -> float:
        w = np.exp(-np.abs(self._difficulty - query) / self.bandwidth)
        w = w / w.sum()
        cum = np.cumsum(w)
        level = min(1.0, (1.0 - self.delta) * (1.0 + 1.0 / w.size))
        idx = int(np.searchsorted(cum, level))
        idx = min(idx, self._scores.size - 1)
        return float(self._scores[idx])

    def interval(self, y_hat: np.ndarray, w: np.ndarray, difficulty: np.ndarray):
        y_hat = np.asarray(y_hat, dtype=np.float64).reshape(-1)
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        difficulty = np.asarray(difficulty, dtype=np.float64).reshape(-1)
        q = np.array([self._weighted_quantile(float(d)) for d in difficulty])
        half = q * w
        return y_hat - half, y_hat + half


class TwoComponentCalibrator:
    """Hierarchical conformal with separate between- and within-engine terms.

    Motivation: for clustered prognostics the scale sigma_i decomposes into an
    engine-level component (how uncertain this engine is overall) and a
    within-engine component (how uncertainty evolves along its trajectory).
    Empirically the two behave in opposite directions: within-engine
    corr(sigma, |err|) > 0 in every subset, between-engine corr < 0. A single
    global quantile therefore cannot be right for both, which is a Simpson's
    paradox.

    Construction (Bonferroni split of delta):
        s_e(i)  = mean sigma over engine of i            (between component)
        q_b     = Quantile_{1-delta_b}( |y - y_hat| / s_e )
        r_i     = max( |y - y_hat| - q_b * s_e(i), 0 )   (residual after between)
        q_w     = Quantile_{1-delta_w}( r_i / sigma_i )
        half_i  = q_b * s_e(i) + q_w * sigma_i

    Coverage is at least 1 - delta_b - delta_w by a union bound.
    """

    def __init__(self, delta: float = 0.1, between_share: float = 0.5) -> None:
        self.delta = float(delta)
        self.between_share = float(between_share)
        self.q_b = None
        self.q_w = None

    @staticmethod
    def engine_mean(values: np.ndarray, units: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        units = np.asarray(units).reshape(-1)
        out = np.zeros_like(values)
        for u in np.unique(units):
            mask = units == u
            out[mask] = values[mask].mean()
        return out

    def fit(self, y, y_hat, sigma, units) -> "TwoComponentCalibrator":
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        y_hat = np.asarray(y_hat, dtype=np.float64).reshape(-1)
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        s_e = self.engine_mean(sigma, units)

        d_b = self.delta * self.between_share
        d_w = self.delta - d_b

        def quantile(vals, level):
            n = vals.size
            lv = min(1.0, (1.0 - level) * (1.0 + 1.0 / n))
            return float(np.quantile(vals, lv, method="higher"))

        self.q_b = quantile(np.abs(y - y_hat) / np.maximum(s_e, 1e-6), d_b)
        resid = np.maximum(np.abs(y - y_hat) - self.q_b * s_e, 0.0)
        self.q_w = quantile(resid / np.maximum(sigma, 1e-6), d_w)
        return self

    def interval(self, y_hat, sigma, units):
        y_hat = np.asarray(y_hat, dtype=np.float64).reshape(-1)
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        s_e = self.engine_mean(sigma, units)
        half = self.q_b * s_e + self.q_w * sigma
        return y_hat - half, y_hat + half
