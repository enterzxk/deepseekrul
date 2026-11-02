"""The one protocol, shared by every table.

An earlier version of this study used three different objects for the same claim: the
calibration scores came from a three-member seed ensemble, the test interval was centred
on a five-fold average, and the metrics were averaged over three seeds.  Each choice is
defensible alone; together they made the "object matching" result unattributable.

This module fixes one definition and every script imports it:

    deployment     P15 = mean over all 5 folds x 3 seeds  (the 15 models that exist)
    calibration    one score per ENGINE, never per (engine, seed)
      cv+            y_i - mu_{-k(i), s(i)}          the single model that excluded i
      matched        y_i - mean_s mu_{-k(i), s}      the 3 clean members; the largest
                                                     clean ensemble the fold structure
                                                     can produce
    normalisation  sigma(yhat) = softplus(a + b*u)*20 + 1, fitted by Gaussian NLL on
                   that arm's own residuals, so the arm cannot borrow another arm's scale
    tails          gamma on a grid that keeps both endpoints finite, so the rank proof
                   applies exactly as stated

Nothing here reads a test label.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C

FOLDS = 5


def fit_scale(yhat, r):
    """sigma(yhat) = softplus(a + b*yhat/125)*20 + 1, fitted by Gaussian NLL."""
    u = np.asarray(yhat, float) / C.MAX_RUL

    def nll(p):
        a, b = p
        s = np.log1p(np.exp(np.clip(a + b * u, -30, 30))) * 20.0 + 1.0
        return float(np.mean(np.log(s) + 0.5 * (r / s) ** 2))

    res = optimize.minimize(nll, x0=[0.0, 0.0], method="Nelder-Mead",
                            options=dict(xatol=1e-4, fatol=1e-8))
    a, b = res.x

    def sigma(v):
        uu = np.asarray(v, float) / C.MAX_RUL
        return np.log1p(np.exp(np.clip(a + b * uu, -30, 30))) * 20.0 + 1.0

    return sigma


def deployed(subset, pred_root=None, tag="A"):
    """The deployed object and the test truth, on the same engines."""
    if pred_root is None:
        pred_root = C.ROOT / "experiments"
    _, tst, layer = C.load_fleet(subset)
    y = layer["y"].astype(float)
    P15 = np.mean([tst[(s, f)][2] for s in C.SEEDS for f in range(FOLDS)], axis=0)
    return P15, y, tst


def _fold_table(dev, fold):
    """(units, y, {seed: pred}) for one fold, aligned on unit id."""
    ref = dev[(C.SEEDS[0], fold)]
    units, y = ref[0], ref[1]
    order = {u: i for i, u in enumerate(units)}
    preds = {}
    for s in C.SEEDS:
        u, _, p = dev[(s, fold)]
        assert set(u) == set(units), f"fold {fold}: seed {s} covers different engines"
        assert np.allclose(dev[(s, fold)][1], y), f"fold {fold}: seeds disagree on y"
        preds[s] = np.array([p[order[x]] for x in units])
    return units, y, preds


def calibration(dev, mode):
    """Score sample for one arm.

    mode = "cv+"      one model per engine (seed cycling so all three are used)
           "matched"  the three clean seed models averaged, one score per engine
    """
    r, centre, unit, fold_of = [], [], [], []
    for f in range(FOLDS):
        units, y, preds = _fold_table(dev, f)
        if mode == "matched":
            p = np.mean([preds[s] for s in C.SEEDS], axis=0)
            r.append(y - p); centre.append(p)
            unit.append(units); fold_of.append(np.full(len(y), f))
        elif mode == "cv+":
            for j, s in enumerate(C.SEEDS):
                p = preds[s]
                r.append(y - p); centre.append(p)
                unit.append(units); fold_of.append(np.full(len(y), f))
        else:
            raise ValueError(mode)
    return (np.concatenate(r), np.concatenate(centre),
            np.concatenate(unit), np.concatenate(fold_of))


def gamma_grid(alpha, m):
    """Admissible tail grid, endpoints excluded.

    gamma = 0 makes the lower endpoint s_(0) = -infinity and gamma = alpha makes the
    upper endpoint s_(m+1) = +infinity.  The implementation clips to the support, so the
    rank proof would not apply as stated at the two ends; restricting the grid to
    [1/(m+1), alpha - 1/(m+1)] keeps both endpoints finite order statistics.
    """
    lo = 1.0 / (m + 1)
    hi = alpha - 1.0 / (m + 1)
    if hi <= lo:
        return np.array([alpha / 2.0])
    return np.linspace(lo, hi, 101)


def pick_gamma(z, alpha, grid=None):
    """Tail split minimising the empirical width, on an interior grid."""
    grid = gamma_grid(alpha, len(z)) if grid is None else grid
    w = [np.quantile(z, min(1.0 - alpha + g, 1.0)) - np.quantile(z, g) for g in grid]
    return float(grid[int(np.argmin(w))])


def arm(subset, mode="matched", normalise=True):
    """Calibration object for one fleet: (dev, scores z, scale function, raw residuals)."""
    dev, _, _ = C.load_fleet(subset)
    r, centre, unit, fold_of = calibration(dev, mode)
    if normalise:
        sigma = fit_scale(centre, r)
        return dict(dev=dev, r=r, z=r / sigma(centre), sigma=sigma, unit=unit, fold=fold_of)
    return dict(dev=dev, r=r, z=r, sigma=lambda v: np.ones_like(np.asarray(v, float)),
                unit=unit, fold=fold_of)


def interval(P, z, level, gamma=None, symmetric=False, sigma=None):
    """Split conformal on the score sample, returning integer-clipped endpoints."""
    alpha = 1.0 - level
    g = alpha / 2.0 if gamma is None else gamma
    sc = None if sigma is None else sigma(P)
    if symmetric:
        lo, hi = C.split_conformal(P, -np.abs(z), np.abs(z), alpha, alpha, scale_test=sc)
    else:
        lo, hi = C.split_conformal(P, z, z, g, alpha - g, scale_test=sc)
    return lo, hi
