"""MDE calculator: minimum detectable difference in RMSE at a given engine count.

Definition used everywhere in the paper:

    MDE(n) = (z_{1-alpha/2} + z_{1-beta}) * sd_i(DeltaRMSE_i) / sqrt(n)

where DeltaRMSE_i is the per-engine RMSE difference between two methods on the same
held-out engines (so engine-level variance is removed), and sd_i is its sample SD.

Two numbers are reported for every pair:
  * paired_mde  — at the engine count actually available (the honest resolution of
                  that experiment);
  * mde_at(n)   — the same formula scaled to another engine count, which is what the
                  "260 engines -> 1.60" style statement means.

This is an extrapolation in n only. It assumes the per-engine SD is stable across
engine sets, which is exactly the assumption a reader should be told about.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

SEEDS = [2027, 2028, 2029]
Z = 1.959964 + 0.841621  # two-sided 5%, power 80%


def load_csv(sub, name, seed, dataset):
    g = sorted(glob.glob(f'/home/yndx/zxk/bearing-rul-pi/experiments/{sub}/{name}_s{seed}/{dataset}/*/matched_cutoff_predictions.csv'))
    if not g:
        raise FileNotFoundError(f'{sub}/{name}_s{seed}/{dataset}')
    return pd.read_csv(g[0])[['unit_id', 'y_true', 'y_pred']]


def per_engine_delta(x, y):
    out = []
    for s in SEEDS:
        a = x(s).set_index('unit_id').sort_index()
        b = y(s).set_index('unit_id').sort_index()
        m = a.join(b, lsuffix='_a', rsuffix='_b', how='inner')
        ea = np.abs(m['y_pred_a'] - m['y_true_a'])
        eb = np.abs(m['y_pred_b'] - m['y_true_b'])
        # per-engine RMSE difference (one matched window per engine -> |e| == RMSE)
        out.append((ea - eb).to_numpy(float))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/dsh/e6/MDE.json')
    a = ap.parse_args()
    jobs = [
        ('FD002 A vs BE', lambda s: load_csv('e3', 'A', s, 'FD002'), lambda s: load_csv('e3', 'BE', s, 'FD002')),
        ('FD002 A vs PE', lambda s: load_csv('e3', 'A', s, 'FD002'), lambda s: load_csv('e3', 'PE', s, 'FD002')),
        ('FD001 A vs BE', lambda s: load_csv('e3_fd001', 'A', s, 'FD001'), lambda s: load_csv('e3_fd001', 'BE', s, 'FD001')),
    ]
    grid = [52, 100, 260, 509, 709, 1460]
    out = {'formula': 'MDE(n) = (1.96 + 0.84) * sd_i(DeltaRMSE_i) / sqrt(n)', 'grid': grid, 'pairs': {}}
    for label, x, y in jobs:
        try:
            d = per_engine_delta(x, y)
        except FileNotFoundError as e:
            print(f'{label:16s} MISSING ({e})')
            out['pairs'][label] = {'error': str(e)}
            continue
        sd = float(np.std(d, ddof=1)); n = len(d)
        row = dict(n_engine_seed=int(n), sd_per_engine=sd,
                   paired_mde=float(Z * sd / np.sqrt(n)),
                   mde_at={str(k): float(Z * sd / np.sqrt(k)) for k in grid},
                   mean_delta=float(d.mean()))
        out['pairs'][label] = row
        print(f'{label:16s} sd={sd:.3f} n={n} paired MDE={row["paired_mde"]:.2f} '
              f'| @260 {row["mde_at"]["260"]:.2f} @509 {row["mde_at"]["509"]:.2f}')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=2)
    print('\n->', a.out)


if __name__ == '__main__':
    main()
