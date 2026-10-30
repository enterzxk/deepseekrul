"""E12 — measurement uncertainty budget for the C-MAPSS comparison.

TIM papers are judged on whether the measurement uncertainty is stated, decomposed
and traceable. This script turns the scattered intervals of sections 4.1-4.3 into
one GUM-style budget:

    u_c = sqrt( sum_i u_i^2 ),      U = k * u_c,  k = 2

Components, each estimated from already-archived runs (no new training):

    u_seed    Type A  scatter across initialisations at fixed fold and architecture
    u_engine  Type A  engine sampling within a fold (bootstrap)
    u_fold    Type A  scatter of the fold-level estimates
    u_arch    Type A  scatter across architectures (filled in by E11)
    u_repro   Type B  re-run determinism (E4: 0.0000 RMSE)

The point of the table is not the total. It is the *ranking*: which component
dominates tells the reader which knob is worth turning. At 52 engines the answer is
already engine sampling; at 260 it is between-fold heterogeneity; the budget says so
in one place.

Run: python scripts/dsh_e12_uncertainty.py
"""
import json
import os

import numpy as np

ROOT = '/home/yndx/zxk/bearing-rul-pi'
OUT = f'{ROOT}/results/dsh/e12/uncertainty_budget.json'


def load(path):
    with open(path) as f:
        return json.load(f)


def budget(name, per_fold, seed_sd, n_seeds, engine_sd, n_eng_per_fold, n_folds,
           arch_sd=None, repro_sd=0.0, k=2.0):
    """Assemble one budget row per estimand."""
    u_seed = seed_sd / np.sqrt(n_seeds) if seed_sd is not None else None
    u_engine = engine_sd / np.sqrt(n_eng_per_fold * n_folds)
    u_fold = float(np.std(per_fold, ddof=1) / np.sqrt(n_folds))
    comps = {'u_seed': u_seed, 'u_engine': u_engine, 'u_fold': u_fold,
             'u_arch': arch_sd, 'u_repro': repro_sd}
    comps = {k: v for k, v in comps.items() if v is not None}
    uc = float(np.sqrt(sum(v ** 2 for v in comps.values())))
    shares = {k: (v ** 2 / uc ** 2) for k, v in comps.items()}
    return dict(estimand=name, n_folds=n_folds, n_engines_per_fold=n_eng_per_fold,
                n_seeds=n_seeds, components=comps, u_combined=uc,
                U_expanded=k * uc, coverage_factor=k,
                variance_shares=shares,
                dominant=max(shares, key=shares.get),
                point_estimate=float(np.mean(per_fold)))


def main():
    e10 = load(f'{ROOT}/results/dsh/e10/E10_cv.json')     # PE - A
    e10b = load(f'{ROOT}/results/dsh/e10/E10b_be_cv.json')  # BE - A
    e9 = load(f'{ROOT}/results/dsh/e9/variance_decomp.json')

    rows = []
    for label, src, seq_key in (('PE - A (bounded time encoding)', e10, 'PE'),
                                ('BE - A (time channel removed)', e10b, 'BE')):
        per_fold = [src['per_fold'][str(f)] for f in range(5)]
        seed_vals = src['per_seed']
        seed_sd = float(np.std(list(seed_vals.values()), ddof=1))
        # within-fold engine scatter: use the FD002 entry of the variance decomposition
        engine_sd = float(e9['FD002']['engine_se'] * np.sqrt(52))  # back out per-engine SD
        rows.append(budget(label, per_fold, seed_sd, 3, engine_sd, 52, 5))

    print(f"{'estimand':32s} {'u_seed':>7s} {'u_eng':>7s} {'u_fold':>7s} "
          f"{'u_c':>7s} {'U':>7s}  dominant")
    for r in rows:
        c = r['components']
        print(f"{r['estimand']:32s} {c.get('u_seed', float('nan')):7.3f} "
              f"{c['u_engine']:7.3f} {c['u_fold']:7.3f} "
              f"{r['u_combined']:7.3f} {r['U_expanded']:7.3f}  "
              f"{r['dominant']} ({100*r['variance_shares'][r['dominant']]:.0f}%)")

    # what would more engines buy? (the question the budget exists to answer)
    print('\nprojection: if the fold count stays at 5, does more engines help?')
    for n_fold_eng in (52, 104, 260):
        e = e9['FD002']['engine_se'] * np.sqrt(52) / np.sqrt(n_fold_eng * 5)
        f = rows[0]['components']['u_fold']
        s = rows[0]['components']['u_seed']
        uc = float(np.sqrt(e ** 2 + f ** 2 + s ** 2))
        print(f'  {n_fold_eng:4d} engines/fold ({n_fold_eng*5:4d} total): '
              f'u_engine={e:.3f} u_fold={f:.3f} -> u_c={uc:.3f} '
              f'(engine share {100*e**2/uc**2:.1f}%)')

    out = dict(
        model='u_c = sqrt(sum u_i^2); U = 2 u_c (k = 2, ~95% coverage)',
        components_documented=dict(
            u_seed='Type A, SD across 3 initialisations at fixed fold and architecture',
            u_engine='Type A, engine sampling within folds (bootstrap SE scaled by 1/sqrt(n))',
            u_fold='Type A, SD of the five fold-level estimates divided by sqrt(5)',
            u_arch='Type A, SD across five architectures (E11), filled in when E11 closes',
            u_repro='Type B, from-scratch re-run (paper/E4_RESULT.md): 0.0000 RMSE'),
        rows=rows,
        note='u_fold dominates at 260 engines; adding engines cannot reduce u_c below the '
             'fold term, which is why section 4.2 recommends reporting fold-level estimates',
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
