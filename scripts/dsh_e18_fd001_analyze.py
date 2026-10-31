"""E18 — FD001 five-fold replication analysis.

Estimator matches `dsh_e16_arch_corrected.py` exactly so the FD002 and FD001 numbers are
comparable:
  * intervals use the single seed every arm has (2027);
  * each fold contributes its 20 held-out engines;
  * clustered bootstrap resamples engines within fold, 100 000 reps, three RNG seeds;
  * delta = RMSE(first) - RMSE(second), negative favours the first arm;
  * TOST at delta = 1.0.

Fold 0 is read from `experiments/e3_fd001` (verified identical partition), folds 1-4 from
`experiments/e18`.

Output: results/dsh/e18/FD001_cv.json
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
SEEDS = [2027, 2028, 2029]
ARMS = ["A", "BE", "PE"]
REPS = 100_000
DELTA = 1.0


def load(arm, seed, fold):
    sub = "e3_fd001" if fold == 0 else "e18"
    pat = f"{EXP}/{sub}/{arm}_s{seed}" + (f"/FD001/{arm}_s{seed}" if fold == 0 else f"_f{fold}/FD001/{arm}_s{seed}") + "/matched_cutoff_predictions.csv"
    g = sorted(glob.glob(pat))
    return pd.read_csv(g[0]).sort_values("unit_id").reset_index(drop=True) if g else None


def metrics(err):
    nasa = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
                nasa=float(np.mean(nasa)), late=float(np.mean(err > 0)),
                severe=float(np.mean(err > 10.0)))


def rmse_of(folds, picks):
    return float(np.sqrt(np.mean(np.concatenate([e[p] for e, p in zip(folds, picks)]) ** 2)))


def contrast(ea, eb, rng_seeds=(9000, 9001, 9002)):
    ident = [slice(None)] * len(ea)
    point = rmse_of(ea, ident) - rmse_of(eb, ident)
    lo90, hi90, lo95, hi95 = [], [], [], []
    for sd in rng_seeds:
        rng = np.random.default_rng(sd)
        acc = np.empty(REPS)
        for r in range(REPS):
            picks = [rng.integers(0, len(e), len(e)) for e in ea]
            acc[r] = rmse_of(ea, picks) - rmse_of(eb, picks)
        lo90.append(np.percentile(acc, 5)); hi90.append(np.percentile(acc, 95))
        lo95.append(np.percentile(acc, 2.5)); hi95.append(np.percentile(acc, 97.5))
    ci90 = [float(np.mean(lo90)), float(np.mean(hi90))]
    ci95 = [float(np.mean(lo95)), float(np.mean(hi95))]
    return dict(delta=point, ci90=ci90, ci95=ci95,
                mc_sd=float(np.std(hi95, ddof=1)),
                excludes_zero=bool(ci95[0] > 0 or ci95[1] < 0),
                tost_equivalent=bool(ci90[0] > -DELTA and ci90[1] < DELTA),
                min_margin_for_equivalence=float(max(abs(ci90[0]), abs(ci90[1]))))


def main() -> None:
    out = {"subset": "FD001", "engines_per_fold": 20, "seed_for_intervals": SEED,
           "reps": REPS, "equivalence_margin": DELTA,
           "sign": "delta = RMSE(first) - RMSE(second); negative favours the first arm",
           "arms": {}, "contrasts": {}, "per_fold": {}, "predictions": {}}

    have = {}
    for arm in ARMS:
        per_fold = [load(arm, SEED, f) for f in FOLDS]
        missing = [f for f, d in zip(FOLDS, per_fold) if d is None]
        if missing:
            print(f"{arm}: missing folds {missing}")
            continue
        have[arm] = [d.signed_error.to_numpy(float) for d in per_fold]

    if len(have) < 3:
        print("incomplete; aborting without writing")
        return

    print("pooled over 100 engines, seed 2027")
    for arm, folds in have.items():
        m = metrics(np.concatenate(folds))
        m["n_engines"] = int(sum(len(e) for e in folds))
        out["arms"][arm] = m
        print(f"  {arm:3s} rmse {m['rmse']:6.3f}  mae {m['mae']:6.3f}  nasa {m['nasa']:5.2f}  "
              f"late {m['late']:.3f}  severe {m['severe']:.3f}")

    # three-seed descriptive means
    print("\nthree-seed means (descriptive)")
    for arm in ARMS:
        runs = [load(arm, s, f) for s in SEEDS for f in FOLDS]
        if not all(r is not None for r in runs):
            continue
        vals = [metrics(r.signed_error.to_numpy(float))["rmse"] for r in runs]
        out["arms"][arm]["three_seed_mean"] = float(np.mean(vals))
        out["arms"][arm]["three_seed_sd"] = float(np.std(vals, ddof=1))
        print(f"  {arm:3s} {np.mean(vals):6.3f} ± {np.std(vals, ddof=1):.3f}  ({len(vals)} runs)")

    print("\nper-fold delta (seed 2027 / 2028 / 2029)")
    for a, b in (("BE", "A"), ("PE", "A")):
        rows = []
        for f in FOLDS:
            row = []
            for s in SEEDS:
                da = load(a, s, f); db = load(b, s, f)
                if da is None or db is None:
                    row.append(None); continue
                ea = da.signed_error.to_numpy(float); eb = db.signed_error.to_numpy(float)
                row.append(float(np.sqrt((ea ** 2).mean()) - np.sqrt((eb ** 2).mean())))
            rows.append(row)
        out["per_fold"][f"{a}_minus_{b}"] = rows
        for f, row in zip(FOLDS, rows):
            print(f"  {a}-{b} fold{f}: " + "  ".join("%+.2f" % x if x is not None else "  --" for x in row))

    print("\npooled contrasts, 100k x 3 RNG")
    for a, b in (("BE", "A"), ("PE", "A"), ("PE", "BE")):
        r = contrast(have[a], have[b])
        out["contrasts"][f"{a}_minus_{b}"] = r
        print(f"  {a} - {b}: {r['delta']:+.3f}  95% CI [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]  "
              f"excludes 0: {r['excludes_zero']}  TOST(d=1): {r['tost_equivalent']}  "
              f"min margin {r['min_margin_for_equivalence']:.2f}")

    # pre-registered predictions
    be = out["contrasts"]["BE_minus_A"]
    pe = out["contrasts"]["PE_minus_A"]
    fold_sd = float(np.std([r[0] for r in out["per_fold"]["BE_minus_A"]], ddof=1))
    out["predictions"] = {
        "P1 BE-A positive": dict(predicted=True, observed=be["delta"], held=bool(be["delta"] > 0)),
        "P2 BE-A CI includes zero": dict(predicted=True, observed=not be["excludes_zero"],
                                          held=bool(not be["excludes_zero"])),
        "P3 |PE-A| < 1.5 and CI includes zero": dict(predicted=True,
                                                     observed=abs(pe["delta"]) < 1.5 and not pe["excludes_zero"],
                                                     held=bool(abs(pe["delta"]) < 1.5 and not pe["excludes_zero"])),
        "P4 fold SD larger than FD002": dict(predicted="> 0.607 (FD002 BE-A)",
                                               observed=fold_sd, held=bool(fold_sd > 0.607)),
    }
    print("\npre-registered predictions:")
    for k, v in out["predictions"].items():
        print(f"  {k:40s} observed {v['observed']!s:20s} held={v['held']}")

    os.makedirs(f"{ROOT}/results/dsh/e18", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/dsh/e18/FD001_cv.json", "w"), indent=2)
    print("\nwrote results/dsh/e18/FD001_cv.json")


if __name__ == "__main__":
    main()
