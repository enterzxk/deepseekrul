"""Three corrections, run together.

A. Power for a paired RMSE difference. Two *different* targets are separated:
     (i)  interval precision: expanded uncertainty U = 2 u_c <= delta
     (ii) power: P(reject H0 at two-sided alpha = 0.05 | true difference = delta) >= 0.80
   Target (i) uses the normal critical value alone; target (ii) also pays for the 80% power
   z-term. Both are checked by a non-parametric paired simulation on the observed per-engine
   influence values, not only by the closed form.

B. Fold decomposition computed on the *paired difference* (BE - A, PE - A), not on one
   model's RMSE. Two distinct quantities are reported and never added:
     within_fold_se  = how much Delta moves if the held-out engines are resampled while the
                       trained model is held fixed  (engine resampling);
     between_fold_sd = how much Delta moves when the training partition changes too
                       (partition change).
   Neither is a "floor" unless the second exceeds the first.

C. Why 24 of the 259 test_FD002 engines were excluded, and what it does to the reported
   number. The rule is seq_len < WINDOW(50) in scripts/dsh_test_verify.py:155. The impact is
   quantified without re-running any model, by comparing the excluded engines' RUL with the
   included engines' and by evaluating a constant predictor on the excluded set.

Output: results/dsh/e19/power_and_folds.json, results/dsh/e19/test_exclusion.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = "/home/yndx/zxk/bearing-rul-pi"
EXP = f"{ROOT}/experiments"
REPS = 20_000
RNG = 9000
ALPHA = 0.05
Z_CRIT = 1.959964
Z_POWER = 0.841621


def load(exp, arm, seed, fold, dataset="FD002"):
    if fold == 0:
        sub = "e3" if dataset == "FD002" else "e3_fd001"
        pat = f"{EXP}/{sub}/{arm}_s{seed}/{dataset}/{arm}_s{seed}/matched_cutoff_predictions.csv"
    else:
        pat = f"{EXP}/{exp}/{arm}_s{seed}_f{fold}/{dataset}/{arm}_s{seed}/matched_cutoff_predictions.csv"
    g = sorted(glob.glob(pat))
    return pd.read_csv(g[0]).sort_values("unit_id").reset_index(drop=True) if g else None


def paired_errors(arm_a, arm_b, exp_a, exp_b, fold, seed=2027, dataset="FD002"):
    a = load(exp_a, arm_a, seed, fold, dataset)
    b = load(exp_b, arm_b, seed, fold, dataset)
    if a is None or b is None:
        return None
    ea = (a.y_pred - a.y_true).to_numpy(float)
    eb = (b.y_pred - b.y_true).to_numpy(float)
    return ea, eb


def influence(ea, eb):
    """Per-engine contribution to (RMSE_a - RMSE_b) by the delta method."""
    ra = np.sqrt(np.mean(ea ** 2))
    rb = np.sqrt(np.mean(eb ** 2))
    z = ea ** 2 / (2 * ra) - eb ** 2 / (2 * rb)
    return z, float(ra), float(rb)


# ------------------------------------------------------------------ A. power
def power_curve(z, grid, rng):
    """Non-parametric paired power simulation.

    z are the per-engine influence values whose mean is the observed Delta. Under H0 the
    distribution is the same but centred on 0. For an alternative delta we add delta to the
    centred draw. The rejection rule is the normal critical value at the simulated SE.
    """
    mu = float(np.mean(z))
    centered = z - mu
    rows = []
    for n in grid:
        draws = rng.choice(centered, size=(REPS, n), replace=True).mean(axis=1)
        se = draws.std(ddof=1)
        # null distribution of the estimator, scaled to this n
        crit = Z_CRIT * se
        # power at the observed effect and at a 5% target
        power_obs = float(np.mean(draws + mu > crit))
        rows.append(dict(n=int(n), se=float(se), observed_delta=mu,
                         power_at_observed=power_obs,
                         mde_closed_form=float((Z_CRIT + Z_POWER) * np.std(z, ddof=1) / np.sqrt(n))))
    return rows


def sample_size_for(z, target_delta):
    """Smallest n for which the simulated power at target_delta reaches 0.80."""
    rng = np.random.default_rng(2026)
    centered = z - z.mean()
    best = None
    for n in [100, 200, 260, 360, 509, 575, 708, 900, 1000, 1258, 1300, 1460, 2000]:
        draws = rng.choice(centered, size=(REPS, n), replace=True).mean(axis=1)
        crit = Z_CRIT * draws.std(ddof=1)
        power = float(np.mean(draws + target_delta > crit))
        if power >= 0.80 and best is None:
            best = dict(n=n, power=power, target_delta=target_delta)
    return best


# ------------------------------------------------------- C. test exclusion
def test_exclusion():
    traj = pd.read_csv(f"{ROOT}/data/CMAPSS/test_FD002.txt", sep=r"\s+", header=None)
    traj = traj.rename(columns={0: "unit", 1: "cycle"})
    rul = pd.read_csv(f"{ROOT}/data/CMAPSS/RUL_FD002.txt", sep=r"\s+", header=None)[0].to_numpy()
    lengths = traj.groupby("unit").cycle.max().to_numpy()
    keep = lengths >= 50
    excl_rul, incl_rul = rul[~keep], rul[keep]
    total_rul = lengths + rul
    out = dict(rule="scripts/dsh_test_verify.py:155  `if seq.shape[0] < WINDOW(50): continue`",
               n_total=int(len(lengths)), n_included=int(keep.sum()), n_excluded=int((~keep).sum()),
               excluded_lengths=[int(x) for x in lengths[~keep]],
               excluded_rul_mean=float(excl_rul.mean()), included_rul_mean=float(incl_rul.mean()),
               excluded_rul_median=float(np.median(excl_rul)),
               included_rul_median=float(np.median(incl_rul)),
               excluded_total_life_mean=float(total_rul[~keep].mean()),
               included_total_life_mean=float(total_rul[keep].mean()))
    # a constant predictor evaluated on the excluded engines, to bound the bias direction
    for cap in (125.0, 100.0):
        err = cap - excl_rul
        pooled = np.concatenate([np.zeros(int(keep.sum())), err])  # placeholder, replaced below
        out[f"constant_{cap:.0f}_rmse_on_excluded"] = float(np.sqrt((err ** 2).mean()))
        out[f"constant_{cap:.0f}_nasa_on_excluded"] = float(
            np.mean(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)))
    return out


def main() -> None:
    out = {"A_power": {}, "B_folds": {}, "C_test_exclusion": {}}
    rng = np.random.default_rng(RNG)

    print("=== A. paired power for an RMSE difference ===")
    for label, na, nb, expa, expb in (("FD002 A vs BE", "A", "BE", "e3", "e3"),
                                      ("FD002 A vs PE", "A", "PE", "e3", "e3")):
        ea, eb = paired_errors(na, nb, expa, expb, 0)
        z, ra, rb = influence(ea, eb)
        grid = [52, 100, 260, 509, 575, 708, 1258]
        rows = power_curve(z, grid, rng)
        sd = float(np.std(z, ddof=1))
        # target 1: expanded uncertainty U = 2 u_c <= 0.70
        u260 = sd / np.sqrt(260)
        n_precision = int(np.ceil((2 * sd / 0.70) ** 2))
        n_power = int(np.ceil(((Z_CRIT + Z_POWER) * sd / 0.70) ** 2))
        sim = sample_size_for(z, 0.70)
        out["A_power"][label] = dict(sd=sd, rows=rows,
                                     u_c_at_260=float(u260),
                                     n_for_halfwidth_070=n_precision,
                                     n_for_80pct_power_070=n_power,
                                     n_for_80pct_power_simulated=sim)
        print(f"\n{label}: sd(z) = {sd:.3f}")
        print(f"  target 1  U = 2 u_c <= 0.70            -> n = {n_precision}")
        print(f"  target 2  80% power at delta = 0.70    -> n = {n_power} (formula), "
              f"{sim['n'] if sim else '>2000'} (simulated, power {sim['power'] if sim else 0:.2f})")
        for r in rows:
            print(f"     n={r['n']:5d}  MDE(80% power) = {r['mde_closed_form']:.2f}   "
                  f"power at observed {r['observed_delta']:+.2f} = {r['power_at_observed']:.2f}")

    print("\n=== B. fold decomposition on the paired difference (not one model's RMSE) ===")
    for label, na, nb, expa, expb in (("BE - A", "BE", "A", "e10b", "e10"),
                                      ("PE - A", "PE", "A", "e10", "e10")):
        per_fold, within = [], []
        for f in range(5):
            pair = paired_errors(na, nb, expa, expb, f)
            if pair is None:
                continue
            ea, eb = pair
            per_fold.append(float(np.sqrt((ea ** 2).mean()) - np.sqrt((eb ** 2).mean())))
            acc = np.empty(REPS // 10)
            for r in range(REPS // 10):
                idx = rng.integers(0, len(ea), len(ea))
                acc[r] = np.sqrt((ea[idx] ** 2).mean()) - np.sqrt((eb[idx] ** 2).mean())
            within.append(float(np.std(acc, ddof=1)))
        between = float(np.std(per_fold, ddof=1))
        out["B_folds"][label] = dict(per_fold_delta=per_fold, within_fold_se=within,
                                     between_fold_sd=between, mean_within_se=float(np.mean(within)),
                                     ratio=between / float(np.mean(within)))
        print(f"{label}: per-fold {['%+.2f' % x for x in per_fold]}")
        print(f"        between-fold SD {between:.3f}  mean within-fold SE {np.mean(within):.3f}  "
              f"ratio {between/np.mean(within):.2f}")

    print("\n=== C. test_FD002 exclusion ===")
    exc = test_exclusion()
    out["C_test_exclusion"] = exc
    print(f"  included {exc['n_included']}, excluded {exc['n_excluded']} of {exc['n_total']}")
    print(f"  excluded lengths: {exc['excluded_lengths']}")
    print(f"  RUL true: excluded mean {exc['excluded_rul_mean']:.1f} / included {exc['included_rul_mean']:.1f}")
    print(f"  total life: excluded mean {exc['excluded_total_life_mean']:.1f} / "
          f"included {exc['included_total_life_mean']:.1f}")
    print(f"  constant-125 RMSE on the excluded engines: {exc['constant_125_rmse_on_excluded']:.2f}")

    os.makedirs(f"{ROOT}/results/dsh/e19", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/dsh/e19/power_and_folds.json", "w"), indent=2)
    json.dump(exc, open(f"{ROOT}/results/dsh/e19/test_exclusion.json", "w"), indent=2)
    print("\nwrote results/dsh/e19/{power_and_folds,test_exclusion}.json")


if __name__ == "__main__":
    main()
