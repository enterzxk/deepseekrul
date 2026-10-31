"""Corrected architecture contrasts + equivalence tests for section 4.2.3.

Two defects in `dsh_e11_arch_analyze.py` are fixed here:

1. The CI slice `gathered[arm][:len(FOLDS)]` selected (f0,s2027), (f0,s2028),
   (f0,s2029), (f1,s2027), (f1,s2028) because the collection loop is fold-outer,
   seed-inner. Fold 0 therefore entered three times and fold 4 never. Every
   interval printed by that script is invalid.
2. No equivalence test was run, so "within a band the members cannot be ordered"
   was asserted from an interval that merely crossed zero. Absence of a
   significant difference is not evidence of equivalence.

Here every contrast uses the same seed (2027), which every arm has, and each fold
contributes exactly 52 engines. Point estimates over three seeds are reported
separately as descriptive numbers and are never used for intervals.

Output: results/dsh/e16/arch_corrected.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = "/home/yndx/zxk/bearing-rul-pi"
EXP = f"{ROOT}/experiments"
FOLDS = list(range(5))
SEED = 2027
REPS = 100_000
RNG = 9000
DELTA = 1.0  # equivalence margin, fixed a priori at the section 4.6 resolution boundary

ARMS = {
    "A": ("e10", True, [2027, 2028, 2029]),
    "tf": ("e11", False, [2027]),
    "gru": ("e11", False, [2027, 2028, 2029]),
    "bilstm": ("e11", False, [2027, 2028, 2029]),
    "tcn": ("e11", False, [2027]),
    "cnn1d": ("e11", False, [2027, 2028, 2029]),
}


def load(arm: str, seed: int, fold: int):
    exp, fold0_is_e3, _ = ARMS[arm]
    path = (f"{EXP}/e3/{arm}_s{seed}/FD002/{arm}_s{seed}/matched_cutoff_predictions.csv"
            if fold == 0 and fold0_is_e3
            else f"{EXP}/{exp}/{arm}_s{seed}_f{fold}/FD002/{arm}_s{seed}/matched_cutoff_predictions.csv")
    hits = glob.glob(path)
    return pd.read_csv(hits[0]).sort_values("unit_id").reset_index(drop=True) if hits else None


def metrics(err: np.ndarray) -> dict:
    nasa = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
                nasa=float(np.mean(nasa)), late=float(np.mean(err > 0)),
                severe=float(np.mean(err > 10.0)))


def rmse_of(fold_errs, picks):
    return float(np.sqrt(np.mean(np.concatenate([e[p] for e, p in zip(fold_errs, picks)]) ** 2)))


def contrast(ea, eb, reps=REPS, rng_seed=RNG):
    """delta = RMSE(a) - RMSE(b), clustered bootstrap resampling engines within fold."""
    identity = [slice(None)] * len(ea)
    point = rmse_of(ea, identity) - rmse_of(eb, identity)
    rng = np.random.default_rng(rng_seed)
    acc = np.empty(reps)
    for r in range(reps):
        picks = [rng.integers(0, len(e), len(e)) for e in ea]
        acc[r] = rmse_of(ea, picks) - rmse_of(eb, picks)
    ci90 = np.percentile(acc, [5, 95])
    ci95 = np.percentile(acc, [2.5, 97.5])
    return dict(delta=point,
                ci90=[float(x) for x in ci90], ci95=[float(x) for x in ci95],
                excludes_zero_95=bool(ci95[0] > 0 or ci95[1] < 0),
                mc_sd=float(np.std(np.percentile(acc, [5, 95]))),
                tost_equivalent=bool(ci90[0] > -DELTA and ci90[1] < DELTA),
                min_margin_for_equivalence=float(max(abs(ci90[0]), abs(ci90[1]))))


def main() -> None:
    common = {}
    for arm, (_, _, seeds) in ARMS.items():
        per_fold = [load(arm, SEED, f) for f in FOLDS]
        if not all(d is not None for d in per_fold):
            print(f"skip {arm}: incomplete at seed {SEED}")
            continue
        common[arm] = [d.signed_error.to_numpy(float) for d in per_fold]

    out = {"seed_for_intervals": SEED, "reps": REPS, "rng": RNG,
           "equivalence_margin": DELTA,
           "sign": "delta = RMSE(first) - RMSE(second); negative means the first arm is better",
           "note": ("corrects dsh_e11_arch_analyze.py, whose CI slice mixed folds and seeds"),
           "arms": {}, "contrasts": {}}

    print("pooled over 260 engines, seed 2027 (common to every arm)")
    print(f"{'arm':8s} {'rmse':>7s} {'mae':>7s} {'nasa':>7s} {'late':>6s} {'severe':>7s}")
    for arm, folds in common.items():
        m = metrics(np.concatenate(folds))
        m["n_engines"] = int(sum(len(e) for e in folds))
        out["arms"][arm] = m
        print(f"{arm:8s} {m['rmse']:7.3f} {m['mae']:7.3f} {m['nasa']:7.3f} "
              f"{m['late']:6.3f} {m['severe']:7.3f}")

    print("\ndescriptive three-seed means (no interval; not used for inference)")
    for arm, (_, _, seeds) in ARMS.items():
        if len(seeds) < 2:
            out.setdefault("three_seed_mean", {})[arm] = None
            continue
        runs = [load(arm, s, f) for s in seeds for f in FOLDS]
        if not all(r is not None for r in runs):
            continue
        per_run = [metrics(r.signed_error.to_numpy(float))["rmse"] for r in runs]
        out.setdefault("three_seed_mean", {})[arm] = float(np.mean(per_run))
        print(f"  {arm:8s} {np.mean(per_run):7.3f}  ({len(per_run)} runs)")

    print(f"\ncontrasts, delta = RMSE(first) - RMSE(second), {REPS} clustered resamples")
    print(f"{'contrast':16s} {'delta':>8s} {'95% CI':>20s} {'zero?':>6s} "
          f"{'TOST(d=1)':>10s} {'min margin':>10s}")
    pairs = [("A", "tf"), ("A", "gru"), ("A", "bilstm"), ("A", "tcn"), ("A", "cnn1d"),
             ("tf", "gru"), ("tf", "bilstm"), ("gru", "bilstm"), ("tcn", "cnn1d")]
    for a, b in pairs:
        if a not in common or b not in common:
            continue
        r = contrast(common[a], common[b])
        out["contrasts"][f"{a}_minus_{b}"] = r
        print(f"{a+' - '+b:16s} {r['delta']:+8.3f} "
              f"[{r['ci95'][0]:+7.3f},{r['ci95'][1]:+7.3f}] "
              f"{'no' if not r['excludes_zero_95'] else 'YES':>6s} "
              f"{'EQUIV' if r['tost_equivalent'] else 'not est':>10s} "
              f"{r['min_margin_for_equivalence']:10.3f}")

    os.makedirs(f"{ROOT}/results/dsh/e16", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/dsh/e16/arch_corrected.json", "w"), indent=2)
    print("\nwrote results/dsh/e16/arch_corrected.json")


if __name__ == "__main__":
    main()
