"""Corrected MDE for an RMSE difference, and a decomposition of the between-fold spread.

Two defects are addressed.

1. `mde.py` used sd(|e_A| - |e_B|), which is the standard error of a difference in **MAE**.
   The paper states an MDE for the difference in **RMSE**. Because RMSE = sqrt(mean(e^2))
   is a non-linear functional, the correct variance comes from the delta method applied to
   the per-engine squared errors:

       Delta = sqrt(mean(e_A^2)) - sqrt(mean(e_B^2))
       z_i   = e_Ai^2 / (2 RMSE_A) - e_Bi^2 / (2 RMSE_B)
       SE(Delta) = sd(z) / sqrt(n)

   Both statistics are reported so the size of the error is visible.

2. Section 4.2.2 claims that between-fold heterogeneity sets a floor that more engines
   cannot lower. That is only true if the fold-to-fold spread exceeds what 52-engine
   sampling alone produces. This script measures the within-fold bootstrap SE of RMSE and
   compares it with the observed spread of the five fold estimates. If they agree, the
   spread is ordinary sampling and shrinks as 1/sqrt(n) like everything else.

Output: results/dsh/e16/mde_corrected.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = "/home/yndx/zxk/bearing-rul-pi"
EXP = f"{ROOT}/experiments"
SEEDS = [2027, 2028, 2029]
Z = 1.959964 + 0.841621  # two-sided 5%, power 80%
REPS = 20_000
GRID = [52, 100, 260, 509, 709, 1460]


def load(sub: str, name: str, seed: int, dataset: str, fold: int | None = None):
    if fold is None:
        pat = f"{EXP}/{sub}/{name}_s{seed}/{dataset}/{name}_s{seed}/matched_cutoff_predictions.csv"
    else:
        base = "e3" if name in ("A", "BE", "PE") and fold == 0 else sub
        pat = (f"{EXP}/e3/{name}_s{seed}/{dataset}/{name}_s{seed}/matched_cutoff_predictions.csv"
               if fold == 0 and name in ("A", "BE", "PE")
               else f"{EXP}/{base}/{name}_s{seed}_f{fold}/{dataset}/{name}_s{seed}/matched_cutoff_predictions.csv")
    g = sorted(glob.glob(pat))
    if not g:
        raise FileNotFoundError(pat)
    return pd.read_csv(g[0])[["unit_id", "y_true", "y_pred"]]


def paired(name_a, name_b, seed, dataset, sub_a="e3", sub_b="e3", fold=None):
    a = load(sub_a, name_a, seed, dataset, fold).sort_values("unit_id").reset_index(drop=True)
    b = load(sub_b, name_b, seed, dataset, fold).sort_values("unit_id").reset_index(drop=True)
    assert (a.unit_id.values == b.unit_id.values).all(), "engine sets differ"
    ea = (a.y_pred - a.y_true).to_numpy(float)
    eb = (b.y_pred - b.y_true).to_numpy(float)
    return ea, eb


def sigma_rmse(ea, eb):
    """Delta-method SD of (RMSE_a - RMSE_b) at the engine level."""
    ra = np.sqrt(np.mean(ea ** 2))
    rb = np.sqrt(np.mean(eb ** 2))
    z = ea ** 2 / (2 * ra) - eb ** 2 / (2 * rb)
    return float(np.std(z, ddof=1)), float(ra), float(rb)


def sigma_mae(ea, eb):
    """The statistic mde.py actually used, kept for comparison."""
    return float(np.std(np.abs(ea) - np.abs(eb), ddof=1))


def main() -> None:
    out = {"formula_rmse": "MDE(n) = (1.96 + 0.84) * sd_i(z_i) / sqrt(n), "
                           "z_i = e_Ai^2/(2 RMSE_A) - e_Bi^2/(2 RMSE_B)",
           "formula_mae": "MDE(n) = (1.96 + 0.84) * sd_i(|e_Ai| - |e_Bi|) / sqrt(n)  <- what mde.py did",
           "grid": GRID, "pairs": {}, "fold_decomposition": {}}

    jobs = [
        ("FD002 A vs BE", "A", "BE", "FD002", "e3", "e3"),
        ("FD002 A vs PE", "A", "PE", "FD002", "e3", "e3"),
        ("FD001 A vs BE", "A", "BE", "FD001", "e3_fd001", "e3_fd001"),
    ]
    for label, na, nb, ds, sa, sb in jobs:
        try:
            ea = np.concatenate([paired(na, nb, s, ds, sa, sb)[0] for s in SEEDS])
            eb = np.concatenate([paired(na, nb, s, ds, sa, sb)[1] for s in SEEDS])
        except FileNotFoundError as exc:
            print(f"skip {label}: {exc}")
            continue
        n = len(ea)
        s_rmse, ra, rb = sigma_rmse(ea, eb)
        s_mae = sigma_mae(ea, eb)
        rec = {"n_obs": n, "sd_rmse_stat": s_rmse, "sd_mae_stat": s_mae,
               "rmse_a": ra, "rmse_b": rb, "delta": ra - rb,
               "mde_at_n": {str(k): Z * s_rmse / np.sqrt(k) for k in GRID},
               "mde_at_n_old": {str(k): Z * s_mae / np.sqrt(k) for k in GRID},
               "inflation": s_rmse / s_mae}
        out["pairs"][label] = rec
        print(f"{label:16s} n={n:4d}  sd(rmse-stat)={s_rmse:7.3f}  sd(mae-stat)={s_mae:7.3f}  "
              f"inflation x{s_rmse/s_mae:.2f}")
        print(f"{'':16s} MDE@52={Z*s_rmse/np.sqrt(52):6.2f}  @260={Z*s_rmse/np.sqrt(260):6.2f}  "
              f"@509={Z*s_rmse/np.sqrt(509):6.2f}  (old @260={Z*s_mae/np.sqrt(260):6.2f})")

    # ---- fold decomposition: is the between-fold spread ordinary sampling? ----
    print("\nfold decomposition for arm A (seed 2027, 52 engines per fold)")
    per_fold, within_se = [], []
    rng = np.random.default_rng(2026)
    for f in range(5):
        d = load("e10", "A", 2027, "FD002", f) if f > 0 else load("e3", "A", 2027, "FD002", 0)
        e = (d.y_pred - d.y_true).to_numpy(float)
        per_fold.append(float(np.sqrt(np.mean(e ** 2))))
        boot = np.empty(REPS)
        for r in range(REPS):
            idx = rng.integers(0, len(e), len(e))
            boot[r] = np.sqrt(np.mean(e[idx] ** 2))
        within_se.append(float(np.std(boot, ddof=1)))
    obs_sd = float(np.std(per_fold, ddof=1))
    mean_within = float(np.mean(within_se))
    out["fold_decomposition"] = {
        "per_fold_rmse": per_fold, "within_fold_bootstrap_se": within_se,
        "observed_between_fold_sd": obs_sd, "mean_within_fold_se": mean_within,
        "ratio": obs_sd / mean_within,
        "interpretation": ("between-fold spread is consistent with 52-engine sampling alone"
                           if abs(obs_sd / mean_within - 1) < 0.5 else
                           "between-fold spread exceeds sampling: an extra training-set component exists"),
    }
    print(f"  per-fold RMSE      {['%.2f' % x for x in per_fold]}")
    print(f"  within-fold SE     {['%.2f' % x for x in within_se]}")
    print(f"  observed between   {obs_sd:.3f}   mean within {mean_within:.3f}   ratio {obs_sd/mean_within:.2f}")
    print(f"  -> {out['fold_decomposition']['interpretation']}")

    os.makedirs(f"{ROOT}/results/dsh/e16", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/dsh/e16/mde_corrected.json", "w"), indent=2)
    print("\nwrote results/dsh/e16/mde_corrected.json")


if __name__ == "__main__":
    main()
