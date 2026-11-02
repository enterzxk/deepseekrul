"""Tail allocation: in-sample, cross-fitted, and equal.

Choosing gamma by minimising the calibration width is a selection on the same scores that
set the endpoints, so the interval that results is no longer the fixed-gamma interval the
guarantee covers. The correction is standard and cheap: choose gamma by cross-validation
*inside* the calibration set, scoring each candidate by the out-of-fold interval score,
then apply the selected gamma to the whole calibration set. This script prices the three
rules on the same engines and, because the benchmark cannot resolve 0.005, repeats the
comparison in the simulation where the true coverage is known.

  gamma = alpha/2          equal tails (the usual convention)
  gamma = in-sample        argmin of the empirical width on the full calibration set
  gamma = cross-fitted     argmin of the out-of-fold interval score, 5 folds

Writes experiments/tim/gamma.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C  # noqa: E402
from run_matrix import LEVELS, best_gamma, fit_scale, scores_for_seed  # noqa: E402
from run_simulation import N_CAL, N_TEST, REPS, draw  # noqa: E402

OUT = C.ROOT / "experiments" / "tim"
GRID = 201


def gamma_crossfit(r, alpha, n_folds=5, seed=3):
    """Cross-validated tail allocation.

    Each candidate gamma is scored by the interval score it would earn on a fold of
    calibration scores it was not fitted on:

        IS(gamma) = (hi - lo) + (2/alpha) * mean[(lo - r)^+ + (r - hi)^+]

    with lo/hi the quantiles of the other folds. The returned gamma is then applied to
    the full calibration sample, so the endpoints are still set by every score, but the
    *choice* no longer saw the scores it is scored against.
    """
    r = np.asarray(r, float)
    idx = np.random.default_rng(seed).permutation(len(r))
    folds = np.array_split(idx, n_folds)
    gs = np.linspace(0.0, alpha, GRID)
    score = np.zeros(GRID)
    for k in range(n_folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        for i, g in enumerate(gs):
            lo = np.quantile(r[tr], g)
            hi = np.quantile(r[tr], min(1.0 - alpha + g, 1.0))
            pen = np.maximum(lo - r[te], 0.0) + np.maximum(r[te] - hi, 0.0)
            score[i] += (hi - lo) + (2.0 / alpha) * float(np.mean(pen))
    return float(gs[int(np.argmin(score))])


def main():
    report = {"levels": list(LEVELS), "subsets": {}, "simulation": {}}

    # ---------------------------------------------------------------- real fleets
    for subset in C.SUBSETS:
        dev, tst, layer = C.load_fleet(subset)
        rows = {}
        for lv in LEVELS:
            alpha = 1.0 - lv
            acc = {k: dict(cov=[], width=[], winkler=[], g=[]) for k in
                   ("equal", "in-sample", "cross-fitted")}
            for seed in C.SEEDS:
                mu = np.stack([tst[(seed, f)][2] for f in range(5)], axis=1)
                yt = tst[(seed, 0)][1]
                pbar = mu.mean(1)
                r_ens, p_ens, _ = scores_for_seed(dev, seed, ensemble=True)
                sig = fit_scale(p_ens, r_ens)
                z, s_tst = r_ens / sig(p_ens), sig(pbar)
                for tag, g in (("equal", alpha / 2.0),
                               ("in-sample", best_gamma(z, alpha)),
                               ("cross-fitted", gamma_crossfit(z, alpha))):
                    lo, hi = C.split_conformal(pbar, z, z, g, alpha - g, scale_test=s_tst)
                    acc[tag]["cov"].append(C.coverage(lo, hi, yt))
                    acc[tag]["width"].append(C.width(lo, hi))
                    acc[tag]["winkler"].append(C.winkler(lo, hi, yt, lv))
                    acc[tag]["g"].append(g)
            for tag, a in acc.items():
                rows[f"{tag}@{lv:.2f}"] = dict(
                    cov=float(np.mean(a["cov"])), width=float(np.mean(a["width"])),
                    winkler=float(np.mean(a["winkler"])),
                    gamma=float(np.mean(a["g"])), n=int(len(layer["y"])))
        report["subsets"][subset] = rows
        print(f"== {subset}")
        for k, v in rows.items():
            print(f"   {k:16s} cov {v['cov']:.3f}  width {v['width']:6.2f}  "
                  f"winkler {v['winkler']:6.2f}  gamma {v['gamma']:.3f}")

    # ---------------------------------------------------------------- simulation
    rng = np.random.default_rng(20261102)
    for n_cal in N_CAL:
        for lv in LEVELS:
            alpha = 1.0 - lv
            acc = {k: [] for k in ("equal", "in-sample", "cross-fitted")}
            for _ in range(REPS):
                yc, pc = draw(rng, n_cal)
                yt, pt = draw(rng, N_TEST)
                r = yc - pc
                for tag, g in (("equal", alpha / 2.0),
                               ("in-sample", best_gamma(r, alpha)),
                               ("cross-fitted", gamma_crossfit(r, alpha))):
                    lo, hi = C.split_conformal(pt, r, r, g, alpha - g)
                    acc[tag].append(C.coverage(lo, hi, yt))
            for tag, v in acc.items():
                report["simulation"][f"{tag}|{n_cal}|{lv:.2f}"] = dict(
                    cov=float(np.mean(v)),
                    cov_se=float(np.std(v, ddof=1) / np.sqrt(REPS)))
            print(f"sim n_cal={n_cal:5d} lv={lv:.2f}  " + "  ".join(
                f"{k} {np.mean(v):.4f}" for k, v in acc.items()))

    (OUT / "gamma.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "gamma.json")


if __name__ == "__main__":
    main()
