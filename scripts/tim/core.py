"""Shared machinery for the TIM study: loaders, conformal constructions, metrics.

Everything downstream (tables, figures, simulation) imports from here so that a single
definition of "coverage" and "width" is used everywhere.

Conventions
-----------
* RUL support is the integer grid 0..125; the target is clipped at 125 by the benchmark.
* A conformal interval is a real interval [lo, hi]; clipping to the support is free
  information and is applied everywhere.
* Coverage of an interval method is the plain empirical frequency (no lattice ambiguity).
* Coverage of a *distribution* (the parametric layer) is measured with randomized PIT,
  because the pmf lives on the lattice and the integer-endpoint rule is biased (see
  `convention_bias`).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
MAX_RUL = 125.0
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SEEDS = (2027, 2028, 2029)


# --------------------------------------------------------------------------- io
def read_csv(path):
    us, ys, ps = [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            us.append(int(r.get("unit_id", r.get("unit"))))
            ys.append(float(r["y_true"]))
            ps.append(float(r["y_pred"]))
    o = np.argsort(us)
    return np.array(us)[o], np.array(ys)[o], np.array(ps)[o]


def load_set(root):
    out = {}
    for f in sorted(Path(root).glob("A_s*_f*.csv")):
        seed, fold = f.stem.split("_")[1], f.stem.split("_")[2]
        out[(int(seed[1:]), int(fold[1:]))] = read_csv(f)
    return out


def load_fleet(subset: str):
    """dev OOF predictions (all cycles, 5 folds x 3 seeds) and per-fold test predictions."""
    dev = load_set(ROOT / "experiments" / "cal" / "dev_preds" / subset)
    tst = load_set(ROOT / "experiments" / "calx" / "fold_test" / subset)
    layer = np.load(ROOT / "experiments" / "cal" / "audit" / f"{subset}_A.npz")
    return dev, tst, layer


# --------------------------------------------------------------------------- metrics
def coverage(lo, hi, y):
    return float(np.mean((lo <= y) & (y <= hi)))


def width(lo, hi):
    return float(np.mean(hi - lo))


def winkler(lo, hi, y, level):
    a = 1.0 - level
    return float(np.mean((hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (float(max(c - h, 0.0)), float(min(c + h, 1.0)))


def pair_mcnemar(lo_a, hi_a, lo_b, hi_b, y):
    """Exact-ish paired comparison of two interval methods on the same engines.

    Returns (n_a_only, n_b_only, p) where `a_only` counts engines covered by A and missed
    by B. The test is the two-sided binomial on the discordant pairs.
    """
    from scipy import stats

    ca = (lo_a <= y) & (y <= hi_a)
    cb = (lo_b <= y) & (y <= hi_b)
    n01 = int(np.sum(ca & ~cb))
    n10 = int(np.sum(~ca & cb))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(n01, n, 0.5).pvalue)
    return n01, n10, p


# --------------------------------------------------------------------------- conformal
def _q_lower(v, q):
    """Lower q-quantile with the finite-sample rank correction (Vovk)."""
    v = np.sort(np.asarray(v, float))
    n = len(v)
    k = int(np.floor(q * (n + 1))) - 1
    return float(v[min(max(k, 0), n - 1)])


def _q_upper(v, q):
    v = np.sort(np.asarray(v, float))
    n = len(v)
    k = int(np.ceil(q * (n + 1))) - 1
    return float(v[min(max(k, 0), n - 1)])


def cvplus(mu_test, r_lo, r_hi, fold_cal, alpha_lo, alpha_hi, scale_test=None):
    """CV+ (Barber et al., 2021) with separate scores for the two arms.

    ``mu_test`` (n_test, K) holds the K fold models' predictions on the test engines.
    ``r_lo``/``r_hi`` are the calibration scores used for the lower and upper endpoint;
    for the asymmetric construction both equal the signed out-of-fold residual
    ``R_i = y_i - mu_{-k(i)}(x_i)``; for the symmetric one they are ``-|R_i|`` and
    ``+|R_i|``. ``fold_cal[i]`` says which fold excluded calibration point i.

    For every calibration point i the model that did not see it is
    ``mu_test[:, fold_cal[i]]``; the endpoint set is that model plus the score.
    """
    n, K = mu_test.shape
    fold_cal = np.asarray(fold_cal, int)
    r_lo = np.asarray(r_lo, float)
    r_hi = np.asarray(r_hi, float)
    if scale_test is not None:
        s = np.asarray(scale_test, float)[:, None]
        A_lo = mu_test[:, fold_cal] + (r_lo * s)
        A_hi = mu_test[:, fold_cal] + (r_hi * s)
    else:
        A_lo = mu_test[:, fold_cal] + r_lo[None, :]
        A_hi = mu_test[:, fold_cal] + r_hi[None, :]
    lo = np.array([_q_lower(A_lo[i], alpha_lo) for i in range(n)])
    hi = np.array([_q_upper(A_hi[i], 1.0 - alpha_hi) for i in range(n)])
    return np.clip(lo, 0, MAX_RUL), np.clip(hi, 0, MAX_RUL)


def cvplus_union(mu_test, r_lo, r_hi, alpha_lo, alpha_hi, scale_test=None):
    """Cross-conformal with every residual paired with every fold model (n_cal x K set).

    More conservative than ``cvplus``; kept as an ablation because the earlier round of
    this project used this form.
    """
    n, K = mu_test.shape
    r_lo = np.asarray(r_lo, float)
    r_hi = np.asarray(r_hi, float)
    if scale_test is not None:
        s = np.asarray(scale_test, float)[:, None, None]
        A_lo = (mu_test[:, :, None] + r_lo[None, None, :] * s).reshape(n, -1)
        A_hi = (mu_test[:, :, None] + r_hi[None, None, :] * s).reshape(n, -1)
    else:
        A_lo = (mu_test[:, :, None] + r_lo[None, None, :]).reshape(n, -1)
        A_hi = (mu_test[:, :, None] + r_hi[None, None, :]).reshape(n, -1)
    lo = np.array([_q_lower(A_lo[i], alpha_lo) for i in range(n)])
    hi = np.array([_q_upper(A_hi[i], 1.0 - alpha_hi) for i in range(n)])
    return np.clip(lo, 0, MAX_RUL), np.clip(hi, 0, MAX_RUL)


def split_conformal(p_test, r_lo, r_hi, alpha_lo, alpha_hi, scale_test=None, rmse_ratio=1.0):
    """Split conformal with separate arm scores; exact finite-sample validity.

    ``rmse_ratio`` rescales the scores when the calibration model and the deployed model
    are not the same object (used only in the mismatch analysis).
    """
    r_lo = np.asarray(r_lo, float) * rmse_ratio
    r_hi = np.asarray(r_hi, float) * rmse_ratio
    a = _q_lower(r_lo, alpha_lo)
    b = _q_upper(r_hi, 1.0 - alpha_hi)
    if scale_test is not None:
        s = np.asarray(scale_test, float)
        lo = np.asarray(p_test, float) + a * s
        hi = np.asarray(p_test, float) + b * s
    else:
        lo = np.asarray(p_test, float) + a
        hi = np.asarray(p_test, float) + b
    return np.clip(lo, 0, MAX_RUL), np.clip(hi, 0, MAX_RUL)


def split_sym(p_test, r, level, scale_test=None):
    """Split conformal with the absolute-residual score, at the level it is meant for.

    The score is |r|, so the (1-alpha) quantile of the scores -- not the (1-alpha/2)
    quantile -- is the one that gives coverage 1-alpha. Passing alpha/2 here is the
    single most common way to make a conformal interval look conservative, so it is
    isolated in one function and used nowhere else.
    """
    alpha = 1.0 - level
    return split_conformal(p_test, -np.abs(r), np.abs(r), alpha, alpha, scale_test=scale_test)


def cvplus_sym(mu_test, r, fold_cal, level, scale_test=None):
    alpha = 1.0 - level
    return cvplus(mu_test, -np.abs(r), np.abs(r), fold_cal, alpha, alpha, scale_test=scale_test)


def arm_lengths(r, alpha_lo, alpha_hi):
    """Lower/upper arm length implied by a score sample and a tail split."""
    r = np.asarray(r, float)
    return -_q_lower(r, alpha_lo), _q_upper(r, 1.0 - alpha_hi)


def symmetry_tax(lo_len, hi_len):
    """Width of the symmetric band that covers the same two arms, vs the asymmetric one."""
    w_asym = lo_len + hi_len
    w_sym = 2.0 * max(lo_len, hi_len)
    return dict(w_asym=float(w_asym), w_sym=float(w_sym),
                tax=float(w_sym - w_asym), rel=float((w_sym - w_asym) / w_asym))


# --------------------------------------------------------------------------- pmf layer
def randomized_pit_cov(p, y, level, nrep=200, seed=7):
    """Coverage of the central interval of a pmf, averaging over the within-bin position."""
    y = np.asarray(y, int)
    cdf = np.cumsum(p, axis=1)
    prev = np.concatenate([np.zeros((len(p), 1)), cdf[:, :-1]], axis=1)
    a = (1.0 - level) / 2.0
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    acc = 0.0
    for _ in range(nrep):
        u = prev[idx, y] + rng.random(len(y)) * p[idx, y]
        acc += float(((u >= a) & (u <= 1.0 - a)).mean())
    return acc / nrep


def shortest_from_pmf(p, level):
    """Shortest interval per row whose pmf mass reaches `level`.

    The central interval (equal tails) is the usual convention, but it is not the
    narrowest interval at that mass unless the pmf is symmetric. On a censored,
    bounded RUL pmf it is not, and the gap is the price of the convention.
    """
    K = p.shape[1]
    n = len(p)
    lo = np.empty(n, int)
    hi = np.empty(n, int)
    for i in range(n):
        cdf = np.concatenate([[0.0], np.cumsum(p[i])])
        best = (K + 1, 0, K - 1)
        for a in range(K):
            b = int(np.searchsorted(cdf, cdf[a] + level - 1e-12)) - 1
            if b < a or b >= K:
                continue
            if (b - a) < best[0]:
                best = (b - a, a, b)
        lo[i], hi[i] = best[1], best[2]
    return lo, hi


def interval_from_pmf(p, level):
    cdf = np.cumsum(p, axis=1)
    a = (1.0 - level) / 2.0
    lo = np.array([int(np.searchsorted(cdf[i], a)) for i in range(len(p))])
    hi = np.array([min(int(np.searchsorted(cdf[i], 1.0 - a)), p.shape[1] - 1) for i in range(len(p))])
    return lo, hi


# --------------------------------------------------------------------------- helpers
def mde(n, p0, alpha=0.05, power=0.80, two_sided=True, direction=-1):
    """Smallest deviation from p0 a binomial test of size n detects with `power`.

    ``direction=-1`` searches the under-coverage side (p1 < p0), which is the side that
    matters for a coverage guarantee. The test rejects when the observed count falls
    outside the exact two-sided acceptance region of Bin(n, p0).
    """
    from scipy import stats

    if n <= 0:
        return float("nan")
    tail = alpha / (2 if two_sided else 1)
    c_lo = int(stats.binom.ppf(tail, n, p0)) - 1
    c_hi = int(stats.binom.isf(tail, n, p0)) + 1

    def power_at(p1):
        if direction < 0:
            return float(stats.binom.cdf(c_lo, n, p1))
        return float(stats.binom.sf(c_hi - 1, n, p1))

    lo, hi = 0.0, p0
    if direction > 0:
        lo, hi = p0, min(p0 + 0.5, 1.0)
    if power_at(hi if direction < 0 else lo) >= power:
        return 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if direction < 0:
            if power_at(mid) >= power:
                lo = mid
            else:
                hi = mid
        else:
            if power_at(mid) >= power:
                hi = mid
            else:
                lo = mid
    return float(abs(p0 - (lo if direction < 0 else hi)))


def bootstrap_units(stat_fn, units, n_boot=2000, seed=11):
    """Block bootstrap over engines (units), the sampling unit of the benchmark."""
    rng = np.random.default_rng(seed)
    idx_by_unit = {u: np.where(units == u)[0] for u in np.unique(units)}
    keys = list(idx_by_unit)
    out = []
    for _ in range(n_boot):
        draw = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([idx_by_unit[u] for u in draw])
        out.append(stat_fn(idx))
    return np.asarray(out)


def summary(arr, q=(2.5, 97.5)):
    arr = np.asarray(arr, float)
    return dict(mean=float(arr.mean()), lo=float(np.percentile(arr, q[0])),
                hi=float(np.percentile(arr, q[1])), sd=float(arr.std(ddof=1)))
