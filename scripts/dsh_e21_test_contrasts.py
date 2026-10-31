"""E21 official-test paired contrasts (BE−A, PE−A) per subset.

Same statistics as the validation-side analyses: delta-method influence functions
z_i = e_A_i^2/(2 RMSE_A) − e_B_i^2/(2 RMSE_B), 100k-rep bootstrap of engine pairs,
3 RNG seeds, 90/95% CIs, TOST equivalence at margin delta = 1.0 RMSE.

Inputs : results/dsh/e21/preds/<subset>_<arm>.csv   (unit, y_pred, y_true, err)
Output : results/dsh/e21/test_contrasts.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRED = os.path.join(ROOT, "results", "dsh", "e21", "preds")
OUT = os.path.join(ROOT, "results", "dsh", "e21", "test_contrasts.json")

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
ARMS = ("A", "BE", "PE")
REPS = 100_000
DELTA = 1.0
RNG_SEEDS = (9000, 9001, 9002)


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


def paired_contrast(err_a: np.ndarray, err_b: np.ndarray):
    """Positive delta = arm A better (larger RMSE for the second arm)."""
    ra, rb = rmse(err_a), rmse(err_b)
    point = rb - ra
    z = err_b ** 2 / (2 * rb) - err_a ** 2 / (2 * ra)
    n = len(z)
    lo90, hi90, lo95, hi95 = [], [], [], []
    for sd in RNG_SEEDS:
        rng = np.random.default_rng(sd)
        acc = np.empty(REPS)
        for r in range(REPS):
            idx = rng.integers(0, n, n)
            acc[r] = rmse(err_b[idx]) - rmse(err_a[idx])
        lo90.append(np.percentile(acc, 5))
        hi90.append(np.percentile(acc, 95))
        lo95.append(np.percentile(acc, 2.5))
        hi95.append(np.percentile(acc, 97.5))
    ci90 = [float(np.mean(lo90)), float(np.mean(hi90))]
    ci95 = [float(np.mean(lo95)), float(np.mean(hi95))]
    return dict(delta=point, ci90=ci90, ci95=ci95,
                mc_sd=float(np.std(hi95, ddof=1)),
                excludes_zero=bool(ci95[0] > 0 or ci95[1] < 0),
                tost_equivalent=bool(ci90[0] > -DELTA and ci90[1] < DELTA),
                min_margin_for_equivalence=float(max(abs(ci90[0]), abs(ci90[1]))))


def main() -> None:
    out = {}
    for sub in SUBSETS:
        frames = {a: pd.read_csv(os.path.join(PRED, f"{sub}_{a}.csv"))
                  for a in ARMS}
        units = {a: set(f["unit"]) for a, f in frames.items()}
        assert all(u == units["A"] for u in units.values()), f"{sub}: unit mismatch"
        errs = {a: f.sort_values("unit")["err"].to_numpy(float) for a, f in frames.items()}
        n = len(errs["A"])
        out[sub] = {
            "n_engines": int(n),
            "rmse": {a: rmse(errs[a]) for a in ARMS},
            "contrasts": {
                "BE_minus_A": paired_contrast(errs["A"], errs["BE"]),
                "PE_minus_A": paired_contrast(errs["A"], errs["PE"]),
            },
        }
        r = out[sub]
        print(f"\n{sub} (n={n})  RMSE A={r['rmse']['A']:.3f} BE={r['rmse']['BE']:.3f} "
              f"PE={r['rmse']['PE']:.3f}")
        for k, v in r["contrasts"].items():
            print(f"  {k}: {v['delta']:+.3f} CI95 [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
                  f"  excl0={v['excludes_zero']}  TOST={v['tost_equivalent']}")

    meta = dict(statistic="RMSE difference, delta-method bootstrap of paired engines",
                reps=REPS, rng_seeds=list(RNG_SEEDS), equivalence_margin=DELTA,
                sign_convention="positive delta = arm B has larger RMSE than arm A "
                                "(A better); BE_minus_A positive = time channel helps",
                note="official test sets, one prediction per engine (last 50-cycle "
                     "window), truth = official RUL files")
    json.dump({**meta, **out}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
