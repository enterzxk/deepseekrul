"""Variance decomposition for the PE - A contrast.

Question: the point estimate is -0.91 RMSE (7.2% of 12.68) with 3/3 seeds
agreeing, yet the engine-clustered CI crosses zero. Which variance dominates,
seed or engine? That decides what to scale: more seeds (cheap) or more
held-out engines (expensive).

Reads the matched-cutoff predictions produced by run_candidate.py.
Writes results/dsh/e9/variance_decomp.json on the server; prints a table.
"""
import json
import glob
import itertools
import os

import numpy as np
import pandas as pd

ROOT = '/home/yndx/zxk/bearing-rul-pi'
SEEDS = [2027, 2028, 2029]
REPS = 20000


def preds(exp, arm, seed, subset='FD002'):
    f = f'{ROOT}/experiments/{exp}/{arm}_s{seed}/{subset}/{arm}_s{seed}/matched_cutoff_predictions.csv'
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    d = pd.read_csv(f).sort_values('unit_id').reset_index(drop=True)
    return d


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def contrast(exp, arm_a, arm_b, seeds, subset='FD002'):
    """Return per-engine error pairs, averaged over seeds, plus per-seed deltas."""
    per_seed = {}
    errs = {}
    for s in seeds:
        a = preds(exp, arm_a, s, subset)
        b = preds(exp, arm_b, s, subset)
        assert (a.unit_id.values == b.unit_id.values).all()
        assert np.allclose(a.y_true.values, b.y_true.values)
        units = a.unit_id.values
        ea = a.signed_error.values.astype(float)
        eb = b.signed_error.values.astype(float)
        per_seed[s] = rmse(ea) - rmse(eb)
        errs[s] = (ea, eb)
    return units, errs, per_seed


def main():
    out = {'contrast': 'PE - A (negative = PE better)', 'reps': REPS,
           'method': 'engine-clustered bootstrap, seeds resampled jointly'}

    for subset, exp in (('FD002', 'e3'), ('FD001', 'e3_fd001')):
        units, errs, per_seed = contrast(exp, 'PE', 'A', SEEDS, subset)
        n_eng = len(units)
        point = float(np.mean(list(per_seed.values())))

        # --- component 1: seed variance (same engines, different init) ---
        seed_sd = float(np.std(list(per_seed.values()), ddof=1))
        seed_se = seed_sd / np.sqrt(len(SEEDS))

        # --- component 2: engine sampling variance, seeds averaged ---
        # average the per-engine squared-error difference across seeds so that
        # the bootstrap sees one observation per engine (seed noise folded in)
        ea = np.mean([errs[s][0] for s in SEEDS], axis=0)
        eb = np.mean([errs[s][1] for s in SEEDS], axis=0)
        rng = np.random.default_rng(20261030)
        acc = np.empty(REPS)
        for r in range(REPS):
            idx = rng.integers(0, n_eng, n_eng)
            acc[r] = rmse(ea[idx]) - rmse(eb[idx])
        eng_se = float(np.std(acc, ddof=1))

        # --- what would N engines give? se scales as 1/sqrt(N) for the
        #     engine component; the seed component is already averaged out.
        proj = {}
        for mult, label in ((2, '2 splits (~2x engines)'), (3, '3 splits (~3x engines)')):
            se = eng_se / np.sqrt(mult)
            proj[label] = dict(n_engines_effective=int(round(n_eng * mult)), se=se,
                               ci=[point - 1.96 * se, point + 1.96 * se],
                               excludes_zero=bool(point + 1.96 * se < 0 or point - 1.96 * se > 0))

        # --- what would more seeds give? (seed component only) ---
        seed_only_se = np.sqrt(seed_se ** 2)  # already seed-averaged for the engine part
        n10 = seed_sd / np.sqrt(10)

        out[subset] = dict(
            n_engines=n_eng, point=point,
            per_seed={str(k): float(v) for k, v in per_seed.items()},
            seed_sd=seed_sd, seed_se_3=float(seed_se), seed_se_10=float(n10),
            engine_se=eng_se,
            variance_share_engine=float(eng_se ** 2 / (eng_se ** 2 + seed_se ** 2)),
            variance_share_seed=float(seed_se ** 2 / (eng_se ** 2 + seed_se ** 2)),
            ci_obs=[point - 1.96 * eng_se, point + 1.96 * eng_se],
            projection=proj,
            relative_effect=float(point / np.mean([rmse(errs[s][0]) for s in SEEDS])),
        )
        d = out[subset]
        print(f'\n=== {subset}  n_engines={n_eng}  point={point:+.3f}  '
              f'({100 * d["relative_effect"]:+.1f}%)')
        print(f'    per-seed delta: ' + ' '.join(f'{v:+.3f}' for v in per_seed.values()))
        print(f'    seed  sd={seed_sd:.3f}  se(3 seeds)={seed_se:.3f}')
        print(f'    engine se={eng_se:.3f}   share engine={d["variance_share_engine"]:.1%}  '
              f'share seed={d["variance_share_seed"]:.1%}')
        print(f'    observed 95% CI: [{d["ci_obs"][0]:+.3f}, {d["ci_obs"][1]:+.3f}]')
        for label, p in proj.items():
            print(f'    if {label:26s} se={p["se"]:.3f}  CI=[{p["ci"][0]:+.3f},{p["ci"][1]:+.3f}]  '
                  f'excludes0={p["excludes_zero"]}')

    os.makedirs(f'{ROOT}/results/dsh/e9', exist_ok=True)
    with open(f'{ROOT}/results/dsh/e9/variance_decomp.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\nwrote results/dsh/e9/variance_decomp.json')


if __name__ == '__main__':
    main()
