"""E10b — 5-fold cross-validated BE vs A on FD002 (does a 2.7 effect survive?).

Pre-registration: paper/E10_CV_PREREGISTRATION.md (sha in every metrics.json).
Estimand: pooled MSE over 260 engines x 3 seeds, then square root; the interval
is a fold-stratified engine-clustered bootstrap with the seeds of an engine kept
together.

Writes results/dsh/e10/E10b_be_cv.json and prints the decision table.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = '/home/yndx/zxk/bearing-rul-pi'
SUBSET = 'FD002'
SEEDS = [2027, 2028, 2029]
FOLDS = [0, 1, 2, 3, 4]
ARMS = ['A', 'BE']
REPS = 100000
RNG_SEEDS = [9000, 9001, 9002, 9003, 9004]
MARGIN_REL = 0.05


def find(arm, seed, fold):
    """fold 0 lives in the archived single-split experiment; folds 1-4 in e10."""
    exp = 'e3' if fold == 0 else ('e10b' if arm == 'BE' else 'e10')
    if fold == 0:
        pat = f'{ROOT}/experiments/e3/{arm}_s{seed}/{SUBSET}/{arm}_s{seed}/matched_cutoff_predictions.csv'
    else:
        pat = f'{ROOT}/experiments/{exp}/{arm}_s{seed}_f{fold}/{SUBSET}/{arm}_s{seed}/matched_cutoff_predictions.csv'
    hits = glob.glob(pat)
    if len(hits) != 1:
        raise FileNotFoundError(f'{pat} -> {len(hits)} hits')
    return hits[0]


def load_fold(fold):
    """Return {arm: array over engines x seeds} plus the engine id order."""
    out = {}
    units = None
    for arm in ARMS:
        per_seed = []
        for s in SEEDS:
            d = pd.read_csv(find(arm, s, fold)).sort_values('unit_id').reset_index(drop=True)
            if units is None:
                units = d.unit_id.to_numpy()
            else:
                assert np.array_equal(units, d.unit_id.to_numpy()), f'engine order mismatch {arm} {s} f{fold}'
            per_seed.append(d.signed_error.to_numpy(dtype=float))
        out[arm] = np.stack(per_seed, axis=1)          # (n_engines, n_seeds)
        out['y_true'] = None
    return units, out


def delta_rmse(err):
    """err: {arm: (n_eng, n_seed)}. Pool the MSE, then take roots.

    Sign convention follows the pre-registration and E6: Delta = RMSE(first) -
    RMSE(second) with first = PE, second = A, so a negative value favours PE.
    """
    a = np.sqrt(np.mean(err['A'] ** 2))
    b = np.sqrt(np.mean(err['BE'] ** 2))
    return b - a, a, b


def main():
    _, _ = None, None
    folds = {}
    for f in FOLDS:
        units, d = load_fold(f)
        folds[f] = dict(units=units, err=d)
        delta, ra, rp = delta_rmse(d)
        print(f'fold {f}: {len(units):3d} engines  RMSE_A={ra:6.3f}  RMSE_BE={rp:6.3f}  delta={delta:+.3f}')

    # coverage check: the five folds must partition the engine set exactly once
    all_units = np.concatenate([folds[f]['units'] for f in FOLDS])
    assert len(all_units) == len(set(all_units)), 'folds overlap'
    n_eng = len(all_units)

    pooled = {arm: np.concatenate([folds[f]['err'][arm] for f in FOLDS], axis=0) for arm in ARMS}
    point, ra, rp = delta_rmse(pooled)
    rel = point / ra
    print(f'\npooled: {n_eng} engines x {len(SEEDS)} seeds = {pooled["A"].size} observations')
    print(f'RMSE_A={ra:.3f}  RMSE_BE={rp:.3f}  delta={point:+.3f}  ({100*rel:+.1f}%)')

    # ---- fold-stratified engine bootstrap, seeds kept together ----
    sizes = [len(folds[f]['units']) for f in FOLDS]
    ups, los, means = [], [], []
    for sd in RNG_SEEDS:
        rng = np.random.default_rng(sd)
        acc = np.empty(REPS)
        for r in range(REPS):
            pick = []
            for f, n in zip(FOLDS, sizes):
                pick.append(rng.integers(0, n, n))
            ea = np.concatenate([folds[f]['err']['A'][p] for f, p in zip(FOLDS, pick)])
            eb = np.concatenate([folds[f]['err']['BE'][p] for f, p in zip(FOLDS, pick)])
            acc[r] = np.sqrt(np.mean(eb ** 2)) - np.sqrt(np.mean(ea ** 2))  # BE - A
        ups.append(float(np.percentile(acc, 97.5)))
        los.append(float(np.percentile(acc, 2.5)))
        means.append(float(acc.mean()))
        print(f'  rng {sd}: 2.5%={los[-1]:+.3f}  97.5%={ups[-1]:+.3f}  mean={means[-1]:+.3f}')

    hi = float(np.mean(ups)); lo = float(np.mean(los))
    mc_hi = float(np.std(ups, ddof=1)); mc_lo = float(np.std(los, ddof=1))

    # E10b rule (paper/E10B_BE_PREREGISTRATION.md section 3): the effect is a
    # *cost*, positive means removing the time channel hurts. REPRODUCED requires
    # the lower bound to clear zero.
    reproduced = bool(lo > 0)
    attenuated = bool(point > 0 and lo <= 0)
    collapsed = bool(point <= 0)
    verdict = ('REPRODUCED' if reproduced else
               'ATTENUATED' if attenuated else 'COLLAPSED')

    per_fold = {str(f): float(delta_rmse(folds[f]['err'])[0]) for f in FOLDS}
    per_seed = {}
    for i, s in enumerate(SEEDS):
        pooled_s = {arm: np.concatenate([folds[f]['err'][arm][:, i:i + 1] for f in FOLDS], axis=0) for arm in ARMS}
        per_seed[str(s)] = float(delta_rmse(pooled_s)[0])

    out = dict(
        prereg='paper/E10B_BE_PREREGISTRATION.md',
        subset=SUBSET, folds=FOLDS, seeds=SEEDS,
        n_engines=int(n_eng), n_observations=int(pooled['A'].size),
        rmse={'A': ra, 'BE': rp}, delta=point, relative=rel,
        ci95=[lo, hi], ci_rng_seeds=RNG_SEEDS, reps=REPS,
        mc_error={'lo': mc_lo, 'hi': mc_hi},
        excludes_zero=bool(hi < 0 or lo > 0),
        per_fold=per_fold, per_seed=per_seed,
        rule=dict(reproduced=reproduced, attenuated=attenuated,
                  collapsed=collapsed, verdict=verdict),
        estimator='pooled MSE over all engines x seeds then sqrt; fold-stratified '
                  'engine bootstrap, seeds of an engine resampled together',
    )
    os.makedirs(f'{ROOT}/results/dsh/e10', exist_ok=True)
    with open(f'{ROOT}/results/dsh/e10/E10b_be_cv.json', 'w') as fh:
        json.dump(out, fh, indent=2)

    print(f'\n95% CI [{lo:+.3f}, {hi:+.3f}]  MC error (hi) {mc_hi:.4f}')
    print(f'excludes zero: {out["excludes_zero"]}   relative {100*rel:+.1f}%')
    print(f'per-fold delta: ' + ' '.join(f'f{k}={v:+.3f}' for k, v in per_fold.items()))
    print(f'per-seed delta: ' + ' '.join(f'{k}={v:+.3f}' for k, v in per_seed.items()))
    print(f'VERDICT: {verdict}')


if __name__ == '__main__':
    main()
