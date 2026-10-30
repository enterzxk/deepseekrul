"""E13 — does the ranking of methods survive a change of metric?

A measurement paper has to say how sensitive its conclusions are to the choice of
instrument. Here the "instrument" has five dials — RMSE, MAE, the NASA asymmetric
score, the late-reporting rate and the severe-late rate — and the question is
whether the ordering of the methods is the same under all five.

If the ordering flips, then "which method is better" is a statement about the
metric as much as about the method, and the paper must say so.

Reads the per-engine predictions already archived (no new training).
Writes results/dsh/e13/metric_sensitivity.json.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = '/home/yndx/zxk/bearing-rul-pi/experiments'
SEEDS = [2027, 2028, 2029]
FOLDS = [0, 1, 2, 3, 4]

# arm -> (experiment dir, is fold0 stored under e3?)
ARMS = {
    'A':      ('e10', True),
    'cnn1d':  ('e11', False),
    'bilstm': ('e11', False),
    'gru':    ('e11', False),
    'tcn':    ('e11', False),
    'tf':     ('e11', False),
}


def load(arm, seed, fold):
    exp, fold0_in_e3 = ARMS[arm]
    if fold == 0 and fold0_in_e3:
        pat = f'{ROOT}/e3/{arm}_s{seed}/FD002/{arm}_s{seed}/matched_cutoff_predictions.csv'
    else:
        pat = f'{ROOT}/{exp}/{arm}_s{seed}_f{fold}/FD002/{arm}_s{seed}/matched_cutoff_predictions.csv'
    g = glob.glob(pat)
    return pd.read_csv(g[0]).sort_values('unit_id').reset_index(drop=True) if g else None


def metrics(e):
    """e = signed errors (pred - true)."""
    return dict(
        rmse=float(np.sqrt(np.mean(e ** 2))),
        mae=float(np.mean(np.abs(e))),
        nasa=float(np.mean(np.where(e < 0, np.exp(-e / 13.0) - 1, np.exp(e / 10.0) - 1))),
        late=float(np.mean(e > 0)),
        severe=float(np.mean(e > 10)),
    )


def main():
    out = {'arms': {}, 'available': [], 'missing': []}
    for arm in ARMS:
        errs = []
        complete = True
        for f in FOLDS:
            for s in SEEDS:
                d = load(arm, s, f)
                if d is None:
                    complete = False
                    break
                errs.append(d.signed_error.to_numpy(float))
            if not complete:
                break
        if not complete:
            out['missing'].append(arm)
            continue
        e = np.concatenate(errs)
        m = metrics(e)
        m['n_obs'] = int(e.size)
        out['arms'][arm] = m
        out['available'].append(arm)

    if not out['available']:
        print('no arm complete yet')
        return

    keys = ['rmse', 'mae', 'nasa', 'late', 'severe']
    print(f"{'arm':8s}" + ''.join(f'{k:>10s}' for k in keys))
    for arm in out['available']:
        m = out['arms'][arm]
        print(f'{arm:8s}' + ''.join(f'{m[k]:10.3f}' for k in keys))

    # rank under each metric (1 = best; for late/severe lower is better too)
    ranks = {}
    for k in keys:
        order = sorted(out['available'], key=lambda a: out['arms'][a][k])
        ranks[k] = {a: i + 1 for i, a in enumerate(order)}
    print('\nrank (1 = best)')
    print(f"{'arm':8s}" + ''.join(f'{k:>10s}' for k in keys))
    for arm in out['available']:
        print(f'{arm:8s}' + ''.join(f'{ranks[k][arm]:10d}' for k in keys))

    # rank agreement between metrics
    agree = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            same = sum(1 for a in out['available'] if ranks[k1][a] == ranks[k2][a])
            agree[f'{k1}_vs_{k2}'] = f'{same}/{len(out["available"])}'

    out['ranks'] = ranks
    out['rank_agreement'] = agree
    out['note'] = ('A ranking that changes between metrics means "which method is '
                   'better" is partly a statement about the metric; the paper reports '
                   'this rather than picking the metric that agrees with its thesis.')
    os.makedirs(f'{ROOT}/../results/dsh/e13', exist_ok=True)
    with open(f'{ROOT}/../results/dsh/e13/metric_sensitivity.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('\nrank agreement:', agree)
    print(f"wrote results/dsh/e13/metric_sensitivity.json  (missing: {out['missing']})")


if __name__ == '__main__':
    main()
