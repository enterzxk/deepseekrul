"""Head-to-head comparison of calibration methods on the same backbone predictions.

The point of this script is to remove the two objections that would otherwise sink the
result: (i) the only baseline was a constant-variance Gaussian, and (ii) methods were
compared at different empirical coverages.

Protocol
- every method sees the SAME development predictions (709 out-of-fold engines), the SAME
  test predictions, and the SAME fitting budget;
- every distributional method gets one width parameter, tuned so that cross-fitted dev
  PICP90 hits 0.90 exactly, after which its test sharpness is comparable;
- interval-only methods (split conformal, CQR) are scored on coverage, width and Winkler
  only -- they do not produce a density, so CRPS/NLL are left blank rather than faked.

Methods
  cens     location-scale MLP, censored (Tobit) likelihood      <- proposed
  trunc    same net, density-on-grid likelihood                  (the v1 behaviour)
  emos_c   affine location-scale, censored                       (EMOS form)
  emos_u   affine location-scale, uncensored density             (plain heteroscedastic)
  homosc   constant sigma, censored
  crude*   empirical error shape on the emos base                (CRUDE, Zelikman 2020)
  isot*    isotonic PIT recalibration of the emos base           (Kuleshov 2018)
  scp      split conformal                                       (interval only)
  nscp     normalised split conformal                            (interval only)
  cqr      conformalised quantile regression                     (interval only)

* CRUDE and isotonic PIT need uncensored targets. On C-MAPSS the cap makes 16-21% of the
  dev labels censored, so both are run two ways: `_exact` treats 125 as the true value,
  `_drop` discards capped engines. Neither is free, and that cost is the honest price of
  using a method that cannot represent an atom.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts" / "cal"))

from layer import (K, MAX_RUL, choose_scale, cover, fit_layer, gen_pmf,  # noqa: E402
                   interval, pmf_np)
from calibrate import load_dev, load_test  # noqa: E402

BANDS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 126)]


# --------------------------------------------------------------------------- helpers
def _params(net, yhat):
    with torch.no_grad():
        mu, sd = net.params(torch.tensor(np.asarray(yhat, dtype=np.float64)))
    return mu.numpy(), sd.numpy()


def pava(y, w=None):
    """Pool adjacent violators, monotone non-decreasing fit."""
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    v, ww = [], []
    for yi, wi in zip(y, w):
        v.append(yi)
        ww.append(wi)
        while len(v) > 1 and v[-2] > v[-1]:
            v2, w2 = v.pop(), ww.pop()
            v1, w1 = v.pop(), ww.pop()
            v.append((v1 * w1 + v2 * w2) / (w1 + w2))
            ww.append(w1 + w2)
    out, i = np.empty_like(y), 0
    for vi, wi in zip(v, ww):
        n = int(round(wi))
        out[i:i + n] = vi
        i += n
    return np.maximum.accumulate(out)


def crps_discrete(p, y):
    """E|X - y| for an integer-valued predictive distribution.

    For integer X this equals the sum of absolute CDF differences, which is *not* the CRPS:
    CRPS additionally subtracts half the expected pairwise distance, a term that depends on
    the distribution but not on y. Reporting this quantity as CRPS would overstate every
    method and, worse, would compare methods whose sharpness differs on an unequal footing.
    Use `crps_proper` when the number is going in a table.
    """
    cdf = np.cumsum(p, axis=1)
    bins = np.arange(K)
    return np.abs(cdf - (bins[None, :] >= np.asarray(y)[:, None])).sum(1)


def crps_proper(p, y):
    """The actual CRPS: E|X - y| - 0.5 E|X - X'|, with E|X - X'| = 2 sum_k F(k)(1 - F(k))."""
    cdf = np.cumsum(p, axis=1)
    bins = np.arange(K)
    l1 = np.abs(cdf - (bins[None, :] >= np.asarray(y)[:, None])).sum(1)
    sharp = (cdf * (1.0 - cdf)).sum(1)
    return l1 - sharp


def nll_censored(p, y):
    idx = np.clip(np.asarray(y, dtype=int), 0, K - 1)
    return -np.log(np.clip(p[np.arange(len(y)), idx], 1e-12, None))


def winkler(lo, hi, y, level=0.90):
    a = 1 - level
    return (hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def widen(p, s):
    """Scale a pmf about its own median: P_s(Y<=e) = P(Y <= m + (e-m)/s)."""
    if s == 1.0:
        return p
    cdf = np.cumsum(p, axis=1)
    m = np.array([np.searchsorted(cdf[i], 0.5) for i in range(len(p))], dtype=np.float64)
    edges = np.arange(1, K, dtype=np.float64)
    x = m[:, None] + (edges[None, :] - m[:, None]) / s
    grid = np.arange(K, dtype=np.float64)
    out = np.zeros((len(p), K), dtype=np.float64)
    for i in range(len(p)):
        Fi = np.interp(x[i], grid, cdf[i], left=0.0, right=1.0)
        row = np.empty(K)
        row[0] = Fi[0]
        row[1:K - 1] = np.diff(Fi)
        row[K - 1] = 1.0 - Fi[-1]
        out[i] = np.clip(row, 0, None)
    return out / out.sum(1, keepdims=True)


def match_coverage(p, y, target=0.90, grid=None):
    """Widen (or sharpen) a pmf until its empirical coverage equals the target.

    Returns the matched pmf, or None when the method cannot reach the target by widening --
    an informative outcome, not a failure, since it means the method's shape cannot trade
    sharpness for coverage at all.
    """
    grid = np.linspace(0.3, 3.0, 55) if grid is None else grid
    if cover(p, y, target) >= target:
        return p, 1.0
    for s in grid:
        ps = widen(p, float(s))
        if cover(ps, y, target) >= target:
            return ps, float(s)
    return None, float("nan")


def width_at_coverage(p, y, target=0.90, grid=None):
    """Width needed to reach `target` coverage, walking the widening path from sharp to flat.

    Coverage is not monotone in the widening factor: past a point the pmf approaches the
    uniform distribution over the grid, whose central 90% excludes the atom, so coverage
    falls again. Sorting the (coverage, width) pairs before interpolating would then return
    the width of an arbitrary branch. Walk the path in order and stop at the FIRST crossing,
    which is the sharpest interval that attains the target -- the same rule used to pick the
    scale, so the two are consistent.
    """
    grid = np.linspace(0.3, 3.0, 55) if grid is None else grid
    prev = None
    for s in grid:
        ps = widen(p, float(s))
        lo, hi = interval(ps, target)
        cov = float(np.mean((lo <= y) & (y <= hi)))
        wid = float(np.mean(hi - lo))
        if cov >= target:
            if prev is None or prev[0] == cov:
                return wid
            c0, w0 = prev
            t = (target - c0) / (cov - c0)
            return float(w0 + t * (wid - w0))
        prev = (cov, wid)
    return float("nan")


def score(p, y):
    lo, hi = interval(p, 0.90)
    lo50, hi50 = interval(p, 0.50)
    lo80, hi80 = interval(p, 0.80)
    out = {
        "crps": float(crps_proper(p, y).mean()),
        "crps_l1": float(crps_discrete(p, y).mean()),
        "nll": float(nll_censored(p, y).mean()),
        "rmse": float(np.sqrt(np.mean(((p * np.arange(K)[None, :]).sum(1) - y) ** 2))),
        "picp50": float(np.mean((lo50 <= y) & (y <= hi50))),
        "picp80": float(np.mean((lo80 <= y) & (y <= hi80))),
        "picp90": float(np.mean((lo <= y) & (y <= hi))),
        "mpiw50": float(np.mean(hi50 - lo50)),
        "mpiw80": float(np.mean(hi80 - lo80)),
        "mpiw90": float(np.mean(hi - lo)),
        "wink90": float(winkler(lo, hi, y).mean()),
        "_per": dict(crps=crps_proper(p, y), nll=nll_censored(p, y),
                     wink=winkler(lo, hi, y), err90=(lo <= y) & (y <= hi),
                     width90=(hi - lo).astype(float), sq=((p * np.arange(K)[None, :]).sum(1) - y) ** 2),
    }
    return out


# --------------------------------------------------------------------------- methods
def fit_bundle(kind, mode):
    def f(dev):
        return fit_layer(dev, kind=kind, mode=mode)
    return f


def make_crude(base_net, train_dev, keep_capped=False):
    """Empirical distribution of standardised residuals, evaluated on a grid."""
    mu, sd = _params(base_net, train_dev["yhat"])
    y = train_dev["y"]
    m = np.ones(len(y), bool) if keep_capped else (y < MAX_RUL)
    z = np.sort((y[m] - mu[m]) / sd[m])
    return z


def km_survival(z, cens):
    """Kaplan-Meier survival curve of Z under right-censoring.

    A capped engine (y == 125) only tells us the true standardised residual is *at least*
    z_obs, so it enters the risk set but never counts as an event. The survivor therefore
    ends on a plateau, and that plateau is precisely the probability mass that belongs in
    the y = 125 atom. Feeding capped engines into a plain empirical CDF instead (what
    `make_crude(..., keep_capped=True)` does) puts a spurious spike at the cap: on C-MAPSS
    16-21% of dev engines sit there, so the spike carries a fifth of the distribution and
    drags the whole shape upward.
    """
    order = np.argsort(z, kind="mergesort")
    zs, cs = z[order], cens[order].astype(bool)
    uniq = np.unique(zs[~cs])
    t_out, surv, s = [], [], 1.0
    for t in uniq:
        at_risk = np.sum(zs >= t)
        d = np.sum((zs == t) & (~cs))
        s *= (1.0 - d / at_risk)
        t_out.append(t)
        surv.append(s)
    return np.asarray(t_out, dtype=np.float64), np.asarray(surv, dtype=np.float64)


def km_eval(t, S, x):
    """Right-continuous step evaluation: S(x) = 1 below the first event time."""
    x = np.asarray(x, dtype=np.float64)
    if len(t) == 0:
        return np.ones_like(x)
    idx = np.searchsorted(t, x, side="right")
    out = np.ones_like(x)
    nz = idx > 0
    out[nz] = S[np.clip(idx[nz] - 1, 0, len(S) - 1)]
    return out


def make_crude_km(base_net, train_dev):
    """CRUDE's empirical error shape, but estimated under right-censoring (KM)."""
    mu, sd = _params(base_net, train_dev["yhat"])
    y = train_dev["y"]
    z = (y - mu) / sd
    return km_survival(z, y >= MAX_RUL)


def crude_km_pmf(par, base_net, yhat, scale=1.0):
    t, S = par
    mu, sd = _params(base_net, yhat)
    sd = sd * scale
    edges = np.arange(1, K)[None, :]
    zz = (edges - mu[:, None]) / sd[:, None]
    Fm = 1.0 - km_eval(t, S, zz.ravel()).reshape(zz.shape)
    p = np.zeros((len(yhat), K))
    p[:, 0] = Fm[:, 0]
    p[:, 1:K - 1] = np.diff(Fm, axis=1)
    p[:, K - 1] = 1.0 - Fm[:, -1]
    p = np.clip(p, 0, None)
    return p / p.sum(1, keepdims=True)


def crude_pmf(z, base_net, yhat, scale=1.0):
    mu, sd = _params(base_net, yhat)
    sd = sd * scale
    edges = np.arange(1, K)[None, :]
    F = (z[None, None, :] <= ((edges[..., None] - mu[:, None, None]) / sd[:, None, None])).mean(-1)
    p = np.zeros((len(yhat), K))
    p[:, 0] = F[:, 0]
    p[:, 1:K - 1] = np.diff(F, axis=1)
    p[:, K - 1] = 1.0 - F[:, -1]
    p = np.clip(p, 0, None)
    return p / p.sum(1, keepdims=True)


def make_isotonic(base_net, train_dev, keep_capped=False):
    """Monotone map from base CDF value to calibrated CDF value (Kuleshov et al. 2018)."""
    from scipy.special import ndtr
    mu, sd = _params(base_net, train_dev["yhat"])
    y = train_dev["y"]
    m = np.ones(len(y), bool) if keep_capped else (y < MAX_RUL)
    u = ndtr((y[m] - mu[m]) / sd[m])
    order = np.argsort(u)
    us = u[order]
    emp = np.arange(1, len(us) + 1) / len(us)
    return pava(emp, np.ones_like(emp)), us


def isotonic_pmf(fit, base_net, yhat):
    from scipy.special import ndtr
    knots, us = fit
    mu, sd = _params(base_net, yhat)
    F = ndtr((np.arange(1, K)[None, :] - mu[:, None]) / sd[:, None])
    Fc = np.interp(F, us, knots, left=0.0, right=1.0)
    p = np.zeros((len(yhat), K))
    p[:, 0] = Fc[:, 0]
    p[:, 1:K - 1] = np.diff(Fc, axis=1)
    p[:, K - 1] = 1.0
    p[:, K - 1] = 1.0 - Fc[:, -1]
    p[:, 0] = np.clip(1.0 - p[:, 1:].sum(1), 0, None)
    p = np.clip(p, 0, None)
    return p / p.sum(1, keepdims=True)


def choose_scale_generic(fit_fn, gen_fn, dev, target=0.90, n_splits=5, seed=0, grid=None):
    """Same rule as layer.choose_scale, but for a method whose generator is not gen_pmf.

    This matters for fairness, not just tidiness. CRUDE and isotonic PIT were previously
    handed the width tuned for the EMOS base, which left them at 0.892 empirical coverage on
    FD001 while EMOS sat at 0.946. CRPS rewards sharpness, so a method that is quietly too
    narrow wins on CRPS for a reason that has nothing to do with its error shape. Every
    method now gets its own width, chosen on cross-fitted dev so that dev coverage hits 0.90,
    and is then evaluated once on test. Test labels are never used to pick a width.
    """
    grid = np.linspace(0.7, 2.5, 37) if grid is None else grid
    rng = np.random.default_rng(seed)
    n = len(dev["y"])
    folds = np.array_split(rng.permutation(n), n_splits)
    fitted = []
    for f in folds:
        mask = np.ones(n, bool)
        mask[f] = False
        fitted.append((fit_fn({k: dev[k][mask] for k in ("yhat", "y")}), f))
    first = None
    for s in grid:
        ps = np.zeros((n, K), dtype=np.float64)
        for net, f in fitted:
            ps[f] = gen_fn(net, dev["yhat"][f], s)
        c = cover(ps, dev["y"], target)
        if first is None:
            first = (float(s), float(c))
        if c >= target:
            return float(s), float(c)
    return first


def split_conformal(dev, level=0.90, sd=None):
    """Fit one residual quantile. Returns a parameter dict, applied separately."""
    res = np.abs(dev["y"] - dev["yhat"])
    q = float(np.quantile(res / sd(dev["yhat"]), level)) if sd else float(np.quantile(res, level))
    return {"q": q, "sd": sd}


def apply_conformal(par, yhat):
    h = par["q"] * par["sd"](yhat) if par["sd"] else np.full(len(yhat), par["q"])
    return np.clip(yhat - h, 0, MAX_RUL), np.clip(yhat + h, 0, MAX_RUL)


def _pl(u, knots):
    """Linear basis in u; monotone by construction once the slope is non-negative."""
    u = np.asarray(u, dtype=np.float64) / MAX_RUL
    return np.stack([np.ones_like(u), u], 1)


def apply_cqr(par, yhat):
    X = np.asarray(_pl(yhat, par["knots"]))
    lo = (X @ par["b0"]) * MAX_RUL - par["q"]
    hi = (X @ par["b1"]) * MAX_RUL + par["q"]
    return np.clip(lo, 0, MAX_RUL), np.clip(hi, 0, MAX_RUL)


def cqr(dev, level=0.90, steps=3000, lr=0.05, knots=6):
    """Split CQR: fit quantiles on half the dev engines, calibrate on the other half."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(dev["y"]))
    half = len(idx) // 2
    a_i, c_i = idx[:half], idx[half:]
    kn = np.linspace(0.1, 0.9, knots)
    feats = lambda u: _pl(u, kn)  # noqa: E731

    qs = []
    for tau in ((1 - level) / 2, 1 - (1 - level) / 2):
        X = torch.tensor(feats(dev["yhat"][a_i]))
        Y = torch.tensor(dev["y"][a_i] / MAX_RUL, dtype=torch.float64)
        a0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        raw = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([a0, raw], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            th = torch.cat([a0, torch.nn.functional.softplus(raw)])
            d = Y - X @ th
            (torch.maximum(tau * d, (tau - 1) * d).mean()).backward()
            opt.step()
        qs.append(torch.cat([a0, torch.nn.functional.softplus(raw)]).detach())
    lo_d = (torch.tensor(feats(dev["yhat"][c_i])) @ qs[0]).numpy() * MAX_RUL
    hi_d = (torch.tensor(feats(dev["yhat"][c_i])) @ qs[1]).numpy() * MAX_RUL
    sc = np.maximum(lo_d - dev["y"][c_i], dev["y"][c_i] - hi_d)
    sc = np.maximum(sc, 0.0)
    n = len(sc)
    q = float(np.quantile(sc, min(1.0, np.ceil((n + 1) * level) / n)))
    return {"b0": qs[0].numpy(), "b1": qs[1].numpy(), "q": q, "knots": kn}


# --------------------------------------------------------------------------- driver
SPEC = {"cens": ("mlp", "censored"), "trunc": ("mlp", "density"),
        "emos_c": ("linear", "censored"), "emos_u": ("linear", "density")}


def score_interval(lo, hi, y):
    level = 0.90
    return {"picp90": float(np.mean((lo <= y) & (y <= hi))),
            "mpiw90": float(np.mean(hi - lo)),
            "wink90": float(winkler(lo, hi, y).mean()),
            "crps": None, "nll": None, "rmse": None,
            "_per": {"err90": (lo <= y) & (y <= hi), "width90": (hi - lo).astype(float),
                     "wink": winkler(lo, hi, y), "crps": None}}


def run(arm, subsets, n_boot=2000, out=None, fit_subsets=None, quiet=False):
    """Fit on `fit_subsets` dev, evaluate on `subsets` test.

    Splitting the two is what makes leave-one-fleet-out possible: fit on three fleets, score
    the fourth. In pooled mode they are the same list. LOSO is the only protocol here that
    answers "does one calibrator transfer to a fleet it has never been fitted on"; pooled
    does not, because it has seen that fleet's dev sample.
    """
    fit_subsets = list(fit_subsets or subsets)
    devs = {s: load_dev(ROOT, s, arm) for s in fit_subsets}
    pooled = {k: np.concatenate([d[k] for d in devs.values()]) for k in ("unit", "yhat", "y")}
    y_dev = pooled["y"]

    # ---- fit everything once, on the same pooled dev sample -------------------------
    nets, times = {}, {}
    for name, (kind, mode) in SPEC.items():
        t = time.time()
        nets[name] = fit_layer(pooled, kind=kind, mode=mode)
        times[name] = time.time() - t

    scales = {}
    for name, (kind, mode) in SPEC.items():
        t = time.time()
        scales[name], _ = choose_scale(lambda d, k=kind, m=mode: fit_layer(d, kind=k, mode=m),
                                       pooled)
        times[name] += time.time() - t

    s0 = float(np.std(y_dev - pooled["yhat"]))
    hom_cov = cover(pmf_np(pooled["yhat"], np.full(len(y_dev), s0), "censored"), y_dev)

    base = nets["emos_c"]
    sd_fn = lambda v: _params(base, v)[1]  # noqa: E731
    z_exact = make_crude(base, pooled, keep_capped=True)
    z_drop = make_crude(base, pooled, keep_capped=False)
    z_km = make_crude_km(base, pooled)
    iso_exact = make_isotonic(base, pooled, keep_capped=True)
    iso_drop = make_isotonic(base, pooled, keep_capped=False)

    # ---- per-method width, each tuned on cross-fitted dev to hit 0.90 ----------------
    # The empirical-shape methods cannot borrow EMOS's width: their own coverage curve is
    # different, and leaving them at EMOS's 0.85 put them at 0.892 on test while EMOS sat at
    # 0.946, which flatters them on CRPS for the wrong reason.
    def _mk_crude_fit(keep):
        def f(d):
            b = fit_layer(d, kind="linear", mode="censored")
            return b, make_crude(b, d, keep_capped=keep)
        return f

    def _mk_crude_gen(pmf_fn):
        def g(pair, yhat, s):
            b, z = pair
            return pmf_fn(z, b, yhat, scale=s)
        return g

    def _km_fit(d):
        b = fit_layer(d, kind="linear", mode="censored")
        return b, make_crude_km(b, d)

    for tag, fit_fn, gen_fn in (
            ("crude_exact", _mk_crude_fit(True), _mk_crude_gen(crude_pmf)),
            ("crude_drop", _mk_crude_fit(False), _mk_crude_gen(crude_pmf)),
            ("crude_km", _km_fit, _mk_crude_gen(crude_km_pmf))):
        t = time.time()
        scales[tag], _ = choose_scale_generic(fit_fn, gen_fn, pooled)
        times[tag] = time.time() - t

    t = time.time()
    sc = split_conformal(pooled)
    nsc = split_conformal(pooled, sd=sd_fn)
    cq = cqr(pooled)
    times["conformal"] = time.time() - t

    # ---- evaluate on each fleet -----------------------------------------------------
    report = {}
    for s in subsets:
        te = load_test(ROOT, s, arm)
        y = te["y"]
        p_te = {
            "cens": gen_pmf(nets["cens"], te["yhat"], scale=scales["cens"]),
            "trunc": gen_pmf(nets["trunc"], te["yhat"], mode="density", scale=scales["trunc"]),
            "emos_c": gen_pmf(nets["emos_c"], te["yhat"], scale=scales["emos_c"]),
            "emos_u": gen_pmf(nets["emos_u"], te["yhat"], mode="density", scale=scales["emos_u"]),
            "homosc": pmf_np(te["yhat"], np.full(len(y), s0), "censored"),
            "crude_exact": crude_pmf(z_exact, base, te["yhat"], scale=scales["crude_exact"]),
            "crude_drop": crude_pmf(z_drop, base, te["yhat"], scale=scales["crude_drop"]),
            "crude_km": crude_km_pmf(z_km, base, te["yhat"], scale=scales["crude_km"]),
            "isot_exact": isotonic_pmf(iso_exact, base, te["yhat"]),
            "isot_drop": isotonic_pmf(iso_drop, base, te["yhat"]),
        }
        rec = {k: score(v, y) for k, v in p_te.items()}
        for k, v in p_te.items():
            rec[k]["sharp90"] = width_at_coverage(v, y, grid=np.linspace(0.5, 3.0, 26))
            pm, sm = match_coverage(v, y, 0.90)
            if pm is None:
                rec[k]["crps_at90"] = None
                rec[k]["widen_to_90"] = None
            else:
                rec[k]["crps_at90"] = float(crps_proper(pm, y).mean())
                rec[k]["widen_to_90"] = sm
        iv = {"scp": apply_conformal(sc, te["yhat"]), "nscp": apply_conformal(nsc, te["yhat"]),
              "cqr": apply_cqr(cq, te["yhat"])}
        for k, (lo_, hi_) in iv.items():
            rec[k] = score_interval(lo_, hi_, y)
            rec[k]["sharp90"] = float(np.mean(hi_ - lo_))

        # ---- conditional coverage by true-RUL band, with Wilson intervals ----------
        bands = []
        for a, b in BANDS:
            m = (y >= a) & (y <= b) if b == 126 else (y >= a) & (y < b)
            if m.sum() == 0:
                continue
            row = {"band": f"[{a},{b})", "n": int(m.sum()), "n_capped": int((y[m] >= MAX_RUL).sum())}
            for name in ("cens", "trunc", "emos_c", "homosc", "crude_exact", "scp"):
                e = rec[name]["_per"]["err90"][m]
                lo95, hi95 = wilson(int(e.sum()), int(m.sum()))
                cell = {"picp": float(e.mean()), "ci": [lo95, hi95]}
                if rec[name]["rmse"] is not None:
                    mu = (p_te[name] * np.arange(K)[None, :]).sum(1)
                    cell["bias"] = float((mu[m] - y[m]).mean())
                else:
                    cell["bias"] = float(np.mean(np.clip(te["yhat"][m], 0, MAX_RUL) - y[m]))
                row[name] = cell
            bands.append(row)

        # ---- paired bootstrap over engines, each method minus `cens` --------------
        rng = np.random.default_rng(7)
        n = len(y)
        bidx = rng.integers(0, n, size=(n_boot, n))
        ref = rec["cens"]["_per"]
        boot = {}
        for name in rec:
            if name == "cens":
                continue
            d = {}
            for key in ("crps", "wink", "width90"):
                a_ = rec[name]["_per"].get(key)
                if a_ is None:
                    d[key] = None
                    continue
                diff = a_ - ref[key]
                vals = diff[bidx].mean(1)
                d[key] = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)),
                          float(diff.mean())]
            boot[name] = d

        report[s] = {
            "n": int(n),
            "methods": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                        for k, v in rec.items()},
            "bands": bands,
            "paired_bootstrap_vs_cens": boot,
        }

    meta = {"arm": arm, "n_dev": int(len(y_dev)), "scales": scales, "homosc_sigma": s0,
            "homosc_dev_cov": hom_cov, "fit_seconds": times, "n_boot": n_boot,
            "n_params": {"cens": 66, "trunc": 66, "emos_c": 4, "emos_u": 4, "homosc": 1,
                         "crude_exact": 0, "crude_drop": 0, "crude_km": 0, "isot": 0,
                         "scp": 1, "nscp": 4, "cqr": 8},
            "protocol": "all methods fitted on the same pooled out-of-fold dev sample; "
                        "every distributional method's width tuned so cross-fitted dev "
                        "PICP90 = 0.90 before touching test",
            "fit_subsets": fit_subsets, "eval_subsets": list(subsets)}
    if quiet:
        return meta, report
    out = Path(out or ROOT / "experiments" / "cal" / f"v2_{arm}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps({"meta": meta, "subsets": report}, indent=1))
    print(f"arm {arm}: wrote {out/'compare.json'}")
    return meta, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A")
    ap.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--loso", action="store_true",
                    help="leave one fleet out: fit on three, score the fourth")
    ap.add_argument("--per-subset", action="store_true",
                    help="fit and score within one fleet; the dev sample is then ~100-250 "
                         "engines instead of 709, which is the only setting where the "
                         "66-parameter layer can be short of data")
    a = ap.parse_args()
    if a.per_subset:
        subs = a.subsets.split(",")
        res = {}
        for tgt in subs:
            meta, rep = run(a.arm, [tgt], a.n_boot, fit_subsets=[tgt], quiet=True)
            res[tgt] = {"meta": meta, "eval": rep[tgt]}
            print(f"  {a.arm} per-subset {tgt}: n_dev={meta['n_dev']} "
                  f"crps(cens)={rep[tgt]['methods']['cens']['crps']:.2f}")
        out = ROOT / "experiments" / "cal" / f"v2_persub_{a.arm}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "per_subset.json").write_text(json.dumps(res, indent=1))
        print(f"arm {a.arm}: wrote {out/'per_subset.json'}")
    elif not a.loso:
        run(a.arm, a.subsets.split(","), a.n_boot)
    else:
        subs = a.subsets.split(",")
        res = {}
        for tgt in subs:
            fit = [s for s in subs if s != tgt]
            meta, rep = run(a.arm, [tgt], a.n_boot, fit_subsets=fit, quiet=True)
            res[tgt] = {"meta": meta, "eval": rep[tgt]}
            print(f"  {a.arm} LOSO <= {tgt}: crps(cens)={rep[tgt]['methods']['cens']['crps']:.2f}")
        out = ROOT / "experiments" / "cal" / f"v2_loso_{a.arm}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "loso.json").write_text(json.dumps(res, indent=1))
        print(f"arm {a.arm}: wrote {out/'loso.json'}")
