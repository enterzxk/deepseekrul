"""Controlled experiment: what, exactly, does matching the calibration object buy?

The earlier protocol compared two things at once.  Its calibration scores were built
from a three-member seed ensemble at the fold that excluded the engine, while its test
interval was centred on a five-fold average and its metrics were averaged over three
seeds -- so the reported width gap mixed the object mismatch, an ensemble-size
mismatch, a different centre, and seed noise.

This script holds everything fixed except the calibration score:

  deployment centre   mean over all 15 models (5 folds x 3 seeds) for every arm
  score transform     z = r / sigma(centre), sigma fitted by Gaussian NLL on that arm's
                      own residuals, same optimiser and same parameterisation
  tail allocation     gamma = alpha/2 for every arm, so the free tail cannot contribute
  test engines        the same 93/235/97/228, one query each
  calibration engines the same 100, the same five folds

and varies only which residuals enter the score sample:

  single-1       one clean model per engine (the model that excluded its fold)
  matched-3      the mean of the three clean seed models at that fold
  single-3x300   the same single model, but all three seeds as separate scores (n=300)
  matched-3x300  the matched score, replicated three times (n=300, same quantiles)

`single` versus `matched` at fixed n is the object effect.  The 3x300 pair repeats the
comparison at the calibration size a three-seed protocol actually uses, so the effect
cannot be attributed to n.  A fifth arm, `matched-3 (5-fold centre)`, keeps the matched
scores but deploys the five-model average the old protocol used, which shows how much of
the previously reported gap was the centre rather than the object.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
from run_matrix import fit_scale

OUT = C.ROOT / "experiments" / "tim"
LEVELS = (0.90, 0.50)


def aligned(dev, seeds, fold):
    """Return (units, y, {seed: pred}) for one fold, aligned on unit id."""
    ref = dev[(seeds[0], fold)]
    units = ref[0]
    order = {u: i for i, u in enumerate(units)}
    y = ref[1]
    preds = {}
    for s in seeds:
        u, _, p = dev[(s, fold)]
        preds[s] = np.array([p[order[x]] for x in units])
        assert np.allclose(np.array([dev[(s, fold)][1][order[x]] for x in units]), y)
    return units, y, preds


def calibration_scores(dev, mode):
    """Score sample for one arm, with the engine that produced each score."""
    seeds = C.SEEDS
    r, centre, unit, fold_of = [], [], [], []
    for f in range(5):
        units, y, preds = aligned(dev, seeds, f)
        if mode == "single-1":
            p = preds[seeds[0]]
            r.append(y - p); centre.append(p); unit.append(units)
            fold_of.append(np.full(len(y), f))
        elif mode == "matched-3":
            p = np.mean([preds[s] for s in seeds], axis=0)
            r.append(y - p); centre.append(p); unit.append(units)
            fold_of.append(np.full(len(y), f))
        elif mode == "single-3x300":
            for s in seeds:
                p = preds[s]
                r.append(y - p); centre.append(p); unit.append(units)
                fold_of.append(np.full(len(y), f))
        elif mode == "matched-3x300":
            p = np.mean([preds[s] for s in seeds], axis=0)
            for _ in seeds:
                r.append(y - p); centre.append(p); unit.append(units)
                fold_of.append(np.full(len(y), f))
        else:
            raise ValueError(mode)
    return (np.concatenate(r), np.concatenate(centre),
            np.concatenate(unit), np.concatenate(fold_of))


def run_subset(subset):
    dev, tst, layer = C.load_fleet(subset)
    y = layer["y"].astype(float)

    # deployment centres, all on the same test engines
    P15 = np.mean([tst[(s, f)][2] for s in C.SEEDS for f in range(5)], axis=0)
    P5 = np.mean([tst[(2027, f)][2] for f in range(5)], axis=0)   # the old protocol's centre

    rows = {}
    for mode in ("single-1", "matched-3", "single-3x300", "matched-3x300"):
        r, centre, _, _ = calibration_scores(dev, mode)
        sigma = fit_scale(centre, r)
        z = r / sigma(centre)
        for lv in LEVELS:
            a2 = (1.0 - lv) / 2.0
            lo, hi = C.split_conformal(P15, z, z, a2, a2, scale_test=sigma(P15))
            rows[f"{mode}|P15|{lv:.2f}"] = dict(
                cov=C.coverage(lo, hi, y), width=C.width(lo, hi),
                winkler=C.winkler(lo, hi, y, lv), n_cal=int(len(r)))
        # the old protocol's centre, matched scores only
        lo, hi = C.split_conformal(P5, z, z, 0.05, 0.05, scale_test=sigma(P5))
        rows[f"{mode}|P5|0.90"] = dict(cov=C.coverage(lo, hi, y), width=C.width(lo, hi),
                                       winkler=C.winkler(lo, hi, y, 0.90), n_cal=int(len(r)))

    # symmetric band on the same matched scores, to confirm the tail is not the lever
    r, centre, _, _ = calibration_scores(dev, "matched-3")
    sigma = fit_scale(centre, r)
    z = r / sigma(centre)
    lo, hi = C.split_sym(P15, z, 0.90, scale_test=sigma(P15))
    rows["matched-3|P15|0.90|sym"] = dict(cov=C.coverage(lo, hi, y), width=C.width(lo, hi),
                                          winkler=C.winkler(lo, hi, y, 0.90), n_cal=int(len(r)))

    # what the deployed objects differ by
    rows["_centres"] = dict(
        rmse_P15=float(np.sqrt(np.mean((y - P15) ** 2))),
        rmse_P5=float(np.sqrt(np.mean((y - P5) ** 2))),
        mean_abs_gap=float(np.mean(np.abs(P15 - P5))),
        n=int(len(y)))
    return rows


def main():
    report = {"levels": list(LEVELS), "subsets": {}}
    for s in C.SUBSETS:
        report["subsets"][s] = run_subset(s)
        c = report["subsets"][s]["_centres"]
        print(f"== {s}  n={c['n']}  RMSE P15 {c['rmse_P15']:.2f}  P5 {c['rmse_P5']:.2f}  "
              f"|P15-P5| {c['mean_abs_gap']:.2f}")
        for k, v in report["subsets"][s].items():
            if k.startswith("_"):
                continue
            print(f"   {k:26s} cov {v['cov']:.3f}  w {v['width']:5.1f}  "
                  f"W {v['winkler']:5.1f}  n_cal {v['n_cal']}")
    (OUT / "object.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "object.json")


if __name__ == "__main__":
    main()
