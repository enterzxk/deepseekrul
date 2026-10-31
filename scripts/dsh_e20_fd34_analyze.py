"""E20 — FD003/FD004 five-fold validation analysis.

Same estimator as E16/E18 so all four subsets are directly comparable:
  seed 2027 for intervals, 100k clustered bootstrap within fold, three RNG seeds,
  delta = RMSE(first) - RMSE(second) (negative favours the first arm), TOST at 1.0.

FD004 fold 4 holds 49 engines, the other folds 50 (Amendment 1 of the pre-registration).

Usage: python scripts/dsh_e20_fd34_analyze.py [--subset FD003|FD004|both]
Output: results/dsh/e20/E20_cv.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = "/home/yndx/zxk/bearing-rul-pi"
EXP = f"{ROOT}/experiments"
SEED = 2027
SEEDS = (2027, 2028, 2029)
FOLDS = (0, 1, 2, 3, 4)
ARMS = ("A", "BE", "PE")
REPS = 100_000
DELTA = 1.0


def load(subset, arm, seed, fold):
    pat = (f"{EXP}/e20/{subset}/{arm}_s{seed}_f{fold}/{subset}/{arm}_s{seed}"
           "/matched_cutoff_predictions.csv")
    g = sorted(glob.glob(pat))
    return pd.read_csv(g[0]).sort_values("unit_id").reset_index(drop=True) if g else None


def metrics(err):
    nasa = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
                nasa=float(np.mean(nasa)), late=float((err > 0).mean()),
                severe=float((err > 10).mean()))


def rmse_of(fold_errs, picks):
    return float(np.sqrt(np.mean(np.concatenate([e[p] for e, p in zip(fold_errs, picks)]) ** 2)))


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
    return dict(delta=point, ci90=ci90, ci95=ci95, mc_sd=float(np.std(hi95, ddof=1)),
                excludes_zero=bool(ci95[0] > 0 or ci95[1] < 0),
                tost_equivalent=bool(ci90[0] > -DELTA and ci90[1] < DELTA),
                min_margin_for_equivalence=float(max(abs(ci90[0]), abs(ci90[1]))))


def analyse(subset) -> dict:
    out = {"subset": subset, "seed_for_intervals": SEED, "reps": REPS,
           "equivalence_margin": DELTA, "arms": {}, "contrasts": {}, "per_fold": {}}
    have = {}
    for arm in ARMS:
        per_fold = [load(subset, arm, SEED, f) for f in FOLDS]
        missing = [f for f, d in zip(FOLDS, per_fold) if d is None]
        if missing:
            print(f"{subset}/{arm}: missing folds {missing} - arm skipped")
            continue
        have[arm] = [d.signed_error.to_numpy(float) for d in per_fold]

    for arm, folds in have.items():
        m = metrics(np.concatenate(folds))
        m["n_engines"] = int(sum(len(e) for e in folds))
        runs = [load(subset, arm, s, f) for s in SEEDS for f in FOLDS]
        vals = [metrics(r.signed_error.to_numpy(float))["rmse"] for r in runs if r is not None]
        if vals:
            m["three_seed_mean"] = float(np.mean(vals))
            m["three_seed_sd"] = float(np.std(vals, ddof=1))
            m["n_runs"] = len(vals)
        out["arms"][arm] = m
        print(f"  {arm:3s} rmse {m['rmse']:6.3f}  nasa {m['nasa']:5.2f}  late {m['late']:.3f}"
              + (f"  3-seed {m.get('three_seed_mean', float('nan')):6.3f}" if vals else ""))

    for a, b in (("BE", "A"), ("PE", "A"), ("PE", "BE")):
        if a not in have or b not in have:
            continue
        r = contrast(have[a], have[b])
        out["contrasts"][f"{a}_minus_{b}"] = r
        print(f"  {a} - {b}: {r['delta']:+.3f}  95% CI [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]  "
              f"excl 0: {r['excludes_zero']}  TOST: {r['tost_equivalent']}  "
              f"min margin {r['min_margin_for_equivalence']:.2f}")

    for a, b in (("BE", "A"), ("PE", "A")):
        if a not in have or b not in have:
            continue
        rows = []
        for f in FOLDS:
            row = []
            for s in SEEDS:
                da, db = load(subset, a, s, f), load(subset, b, s, f)
                if da is None or db is None:
                    row.append(None); continue
                ea = da.signed_error.to_numpy(float); eb = db.signed_error.to_numpy(float)
                row.append(float(np.sqrt((ea ** 2).mean()) - np.sqrt((eb ** 2).mean())))
            rows.append(row)
        out["per_fold"][f"{a}_minus_{b}"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="both", choices=["FD003", "FD004", "both"])
    a = ap.parse_args()
    subsets = ["FD003", "FD004"] if a.subset == "both" else [a.subset]
    out = {}
    for s in subsets:
        print(f"\n=== {s} (pooled over held-out engines, seed 2027) ===")
        out[s] = analyse(s)
    os.makedirs(f"{ROOT}/results/dsh/e20", exist_ok=True)
    path = f"{ROOT}/results/dsh/e20/E20_cv.json"
    existing = {}
    if os.path.exists(path):
        try:
            existing = json.load(open(path))
        except Exception:
            existing = {}
    existing.update(out)
    json.dump(existing, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
