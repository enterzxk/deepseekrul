"""Final construction matrix for the TIM study.

Constructions, all evaluated on the same engines at the same nominal levels:

  layer       frozen 66-parameter censored parametric layer (the reference method)
  SC-sym      split conformal, one fold model, |r| score
  SC-asym     split conformal, one fold model, signed score, equal tails
  CV+-sym     cross-conformal (Barber et al.), |r| score
  CV+-asym    cross-conformal, signed score, equal tails
  EAC-eq      ensemble-aware: centre = deployed K-model mean, scores = seed-ensemble OOF
              residuals, equal tails
  EAC         same, tail allocation chosen to minimise empirical width
  EAC-norm    EAC with the scores divided by a fitted variance function

The distinction that matters in the paper: CV+ pairs each calibration residual with the
single fold model that excluded it, which is a different object from the ensemble that
is actually deployed. EAC calibrates the deployed object.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C

LEVELS = tuple(float(x) for x in
                os.environ.get("RUL_LEVELS", "0.90,0.50").split(","))
TAG = os.environ.get("RUL_PRED_TAG", "A")
OUT = C.ROOT / "experiments" / "tim"
OUT.mkdir(parents=True, exist_ok=True)


def _out_name():
    """matrix.json for the default run; anything else is suffixed so nothing is clobbered."""
    base = "matrix" if TAG == "A" else f"matrix_{TAG}"
    if LEVELS == (0.90, 0.50):
        return OUT / f"{base}.json"
    lv = "_".join(f"{x:.2f}".replace("0.", "") for x in LEVELS)
    return OUT / f"{base}_lv{lv}.json"


def fit_scale(yhat, r):
    """sigma(yhat) = softplus(a + b*u)*20 + 1, u = yhat/125, fitted by Gaussian NLL."""
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


def best_gamma(r, alpha, grid=201):
    """Tail split that minimises the empirical width at miscoverage `alpha`."""
    gs = np.linspace(0.0, alpha, grid)
    w = [np.quantile(r, min(1.0 - alpha + g, 1.0)) - np.quantile(r, g) for g in gs]
    return float(gs[int(np.argmin(w))])


def scores_for_seed(dev, seed, ensemble=True):
    """Calibration scores and centres.

    ensemble=True  -> centre is the mean over the three seeds of the fold model that
                      excluded the engine (a 3-model ensemble; every member is clean).
    ensemble=False -> centre is the single fold model that excluded it.
    """
    r, p, fold = [], [], []
    seeds = C.SEEDS if ensemble else (seed,)
    for f in range(5):
        yc = dev[(seed, f)][1]
        pc = np.mean([dev[(s, f)][2] for s in seeds], axis=0)
        r.append(yc - pc)
        p.append(pc)
        fold.append(np.full(len(yc), f))
    return np.concatenate(r), np.concatenate(p), np.concatenate(fold)


def eval_intervals(lo, hi, y, lv):
    return dict(cov=C.coverage(lo, hi, y), width=C.width(lo, hi),
                winkler=C.winkler(lo, hi, y, lv),
                cov_lo=C.wilson(int(round(C.coverage(lo, hi, y) * len(y))), len(y))[0],
                cov_hi=C.wilson(int(round(C.coverage(lo, hi, y) * len(y))), len(y))[1])


def run_subset(subset):
    dev, tst, layer = C.load_fleet(subset)
    y = layer["y"].astype(float)
    n = len(y)
    rows = {}

    # ---- frozen layer
    # Coverage is reported on the integer-endpoint rule for every construction, so that
    # one table compares like with like.  The mass-based figure (`cov_mass`) is carried
    # alongside as a diagnostic, never as the headline: for two intervals built to the
    # same nominal level it is equal by construction, so it cannot rank them.
    for lv in LEVELS:
        lo_i, hi_i = C.interval_from_pmf(layer["p_cens"], lv)
        lo_s, hi_s = C.shortest_from_pmf(layer["p_cens"], lv)
        for tag, (lo, hi) in (("layer-short", (lo_s, hi_s)), ("layer", (lo_i, hi_i))):
            lo = lo.astype(float)
            hi = hi.astype(float)
            rows[f"{tag}@{lv:.2f}"] = dict(
                cov=C.coverage(lo, hi, y),
                cov_mass=C.interval_mass(layer["p_cens"], lo.astype(int), hi.astype(int)),
                width=C.width(lo, hi),
                winkler=C.winkler(lo, hi, y, lv),
                cov_lo=C.wilson(int(round(C.coverage(lo, hi, y) * n)), n)[0],
                cov_hi=C.wilson(int(round(C.coverage(lo, hi, y) * n)), n)[1],
                n=n)
        rows[f"layer-pit@{lv:.2f}"] = dict(
            cov=C.randomized_pit_cov(layer["p_cens"], y.astype(int), lv), n=n)

    acc = {}
    gamma_log = {}

    def add(key, lo, hi, lv, yy):
        acc.setdefault(key, []).append(eval_intervals(lo, hi, yy, lv))

    for seed in C.SEEDS:
        mu = np.stack([tst[(seed, f)][2] for f in range(5)], axis=1)
        yt = tst[(seed, 0)][1]
        pbar = mu.mean(1)

        r_ens, p_ens, fold_ens = scores_for_seed(dev, seed, ensemble=True)
        r_sing, p_sing, fold_sing = scores_for_seed(dev, seed, ensemble=False)
        sigma = fit_scale(p_ens, r_ens)
        s_tst = sigma(pbar)
        s_cal = sigma(p_ens)

        for lv in LEVELS:
            alpha = 1.0 - lv
            a2 = alpha / 2.0
            abs_ens = np.abs(r_ens)

            # split conformal, single fold model (exact-validity reference)
            for tag, signed in (("SC-sym", False), ("SC-asym", True)):
                cls, wis, wks = [], [], []
                for f in range(5):
                    yc = dev[(seed, f)][1]
                    pc = dev[(seed, f)][2]
                    rc = yc - pc
                    _, yt_f, pt_f = tst[(seed, f)]
                    if signed:
                        lo, hi = C.split_conformal(pt_f, rc, rc, a2, a2)
                    else:
                        lo, hi = C.split_sym(pt_f, rc, lv)
                    cls.append(C.coverage(lo, hi, yt_f))
                    wis.append(C.width(lo, hi))
                    wks.append(C.winkler(lo, hi, yt_f, lv))
                acc.setdefault(f"{tag}@{lv:.2f}", []).append(
                    dict(cov=float(np.mean(cls)), width=float(np.mean(wis)),
                         winkler=float(np.mean(wks)), cov_lo=None, cov_hi=None))

            # cross-conformal on the single-model pairing
            lo, hi = C.cvplus_sym(mu, r_sing, fold_sing, lv)
            add(f"CV+-sym@{lv:.2f}", lo, hi, lv, yt)
            lo, hi = C.cvplus(mu, r_sing, r_sing, fold_sing, a2, a2)
            add(f"CV+-asym@{lv:.2f}", lo, hi, lv, yt)

            # ensemble-aware, symmetric (normalised, so only the symmetry assumption differs)
            z = r_ens / s_cal
            lo, hi = C.split_sym(pbar, z, lv, scale_test=s_tst)
            add(f"EAC-sym@{lv:.2f}", lo, hi, lv, yt)

            # ensemble-aware, asymmetric, equal tails
            lo, hi = C.split_conformal(pbar, z, z, a2, a2, scale_test=s_tst)
            add(f"EAC-eq@{lv:.2f}", lo, hi, lv, yt)

            # EAC: tail allocation chosen on the normalised scores to minimise width
            g = best_gamma(z, alpha)
            lo, hi = C.split_conformal(pbar, z, z, g, alpha - g, scale_test=s_tst)
            add(f"EAC@{lv:.2f}", lo, hi, lv, yt)
            gamma_log.setdefault(lv, []).append(g)

    for key, vals in acc.items():
        if not vals:
            continue
        agg = {"n": n}
        for k in ("cov", "width", "winkler"):
            xs = [v[k] for v in vals if v[k] is not None and not (isinstance(v[k], float) and np.isnan(v[k]))]
            agg[k] = float(np.mean(xs)) if xs else None
        for k in ("cov", "width"):
            xs = [v[k] for v in vals]
            agg[f"{k}_sd"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        rows[key] = agg
    for lv, gs in gamma_log.items():
        rows[f"gamma@{lv:.2f}"] = dict(gamma=float(np.mean(gs)), n=n)
    return rows


def main():
    report = {"levels": list(LEVELS), "subsets": {}}
    for subset in C.SUBSETS:
        rows = run_subset(subset)
        report["subsets"][subset] = rows
        print(f"== {subset} (n={rows[f'layer@{LEVELS[0]:.2f}']['n']})")
        for k, v in rows.items():
            w = f"{v['width']:5.1f}" if v.get("width") is not None else "  -  "
            c = f"{v['cov']:.3f}" if v.get("cov") is not None else "  -  "
            print(f"   {k:14s} cov {c}  w {w}")
    dst = _out_name()
    dst.write_text(json.dumps(report, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
