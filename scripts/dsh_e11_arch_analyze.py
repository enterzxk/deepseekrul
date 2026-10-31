"""E11 — architecture comparison on the identical 5-fold protocol.

Seven arms, same folds, same split seed, same budget:
    A        candidate (sensors + EMA + cycle), 3 seeds
    BE       candidate without the cycle channel, 3 seeds
    cnn1d    1-D CNN, 3 seeds
    bilstm   BiLSTM, 3 seeds
    gru      GRU + attention, 3 seeds
    tcn      dilated TCN, 1 seed  (Amendment 1)
    tf       vanilla Transformer, 1 seed (Amendment 1)

Units of analysis: each engine is averaged over its available seeds first, then the 260
engine-level values are pooled. This keeps a 1-seed arm comparable with a 3-seed arm
(Amendment 1 explains why the pooled-observation route would not be).

Interval: fold-stratified engine bootstrap, 100 000 reps, RNG 9000-9004.

Writes results/dsh/e11/E11_arch.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = '/home/yndx/zxk/bearing-rul-pi/experiments'
SEEDS = [2027, 2028, 2029]
FOLDS = range(5)
REPS = 100000
RNG_SEEDS = [9000, 9001, 9002, 9003, 9004]

# arm -> (experiment dir, seeds available, fold0 stored under e3?)
ARMS = {
    'A':      ('e10', SEEDS, True),
    'BE':     ('e10b', SEEDS, False),
    'cnn1d':  ('e11', SEEDS, False),
    'bilstm': ('e11', SEEDS, False),
    'gru':    ('e11', SEEDS, False),
    'tcn':    ('e11', [2027], False),
    'tf':     ('e11', [2027], False),
}


def load(arm, seed, fold):
    exp, _, f0 = ARMS[arm]
    if fold == 0 and f0:
        pat = f'{ROOT}/e3/{arm}_s{seed}/FD002/{arm}_s{seed}/matched_cutoff_predictions.csv'
    else:
        pat = f'{ROOT}/{exp}/{arm}_s{seed}_f{fold}/FD002/{arm}_s{seed}/matched_cutoff_predictions.csv'
    g = glob.glob(pat)
    return pd.read_csv(g[0]).sort_values('unit_id').reset_index(drop=True) if g else None


def metrics(e):
    return dict(
        rmse=float(np.sqrt(np.mean(e ** 2))),
        mae=float(np.mean(np.abs(e))),
        nasa=float(np.mean(np.where(e < 0, np.exp(-e / 13.0) - 1, np.exp(e / 10.0) - 1))),
        late=float(np.mean(e > 0)),
        severe=float(np.mean(e > 10)),
    )


def main():
    out = {'protocol': 'FD002, 5-fold CV, 260 engines held out once each',
           'unit_of_analysis': 'per-engine mean over available seeds, then pooled',
           'arms': {}, 'missing': []}

    gathered = {}
    for arm, (_, seeds, _) in ARMS.items():
        per_run = []          # one error vector per (fold, seed); no cross-seed averaging
        complete = True
        for f in FOLDS:
            for s in seeds:
                d = load(arm, s, f)
                if d is None:
                    complete = False
                    break
                per_run.append(d.signed_error.to_numpy(float))
            if not complete:
                break
        if not complete:
            out['missing'].append(arm)
            continue
        gathered[arm] = per_run

    # point estimate: mean over (fold, seed) of the per-run metric, so that a 3-seed arm
    # gets no ensemble advantage over a 1-seed arm
    for arm, runs in gathered.items():
        per_metric = [metrics(e) for e in runs]
        m = {k: float(np.mean([x[k] for x in per_metric])) for k in per_metric[0]}
        m['n_runs'] = len(runs)
        m['n_engines'] = int(sum(len(e) for e in runs[-len(FOLDS):]))
        m['seeds'] = len(ARMS[arm][1])
        out['arms'][arm] = m

    keys = ['rmse', 'mae', 'nasa', 'late', 'severe']
    print(f"{'arm':8s}{'seeds':>6s}" + ''.join(f'{k:>9s}' for k in keys))
    for arm in ARMS:
        if arm in out['arms']:
            m = out['arms'][arm]
            print(f"{arm:8s}{m['seeds']:>6d}" + ''.join(f'{m[k]:9.3f}' for k in keys))

    # bootstrap each arm against A
    print('\nvs A (positive = arm worse):')
    contrasts = {}
    # CI uses seed 2027 only, because every arm has it (Amendment 1)
    if 'A' in gathered:
        a = np.concatenate(gathered['A'][:len(FOLDS)])
        for arm in ARMS:
            if arm == 'A' or arm not in gathered:
                continue
            b = np.concatenate(gathered[arm][:len(FOLDS)])
            point = float(np.sqrt(np.mean(b ** 2)) - np.sqrt(np.mean(a ** 2)))
            rng = np.random.default_rng(9000)
            acc = np.empty(20000)
            for r in range(20000):
                idx = np.concatenate([f * 52 + rng.integers(0, 52, 52) for f in range(5)])
                acc[r] = np.sqrt(np.mean(b[idx] ** 2)) - np.sqrt(np.mean(a[idx] ** 2))
            lo, hi = np.percentile(acc, [2.5, 97.5])
            contrasts[arm] = dict(delta=point, ci=[float(lo), float(hi)],
                                  excludes_zero=bool(lo > 0 or hi < 0))
            print(f'  {arm:8s} {point:+7.3f}  [{lo:+7.3f},{hi:+7.3f}]  '
                  f'excludes0={contrasts[arm]["excludes_zero"]}')

    rmses = [out['arms'][a]['rmse'] for a in out['arms']]
    out['spread'] = dict(min=float(min(rmses)), max=float(max(rmses)),
                         range=float(max(rmses) - min(rmses)))
    out['contrasts_vs_A'] = contrasts
    out['verdict'] = ('DISPERSED' if any(v['excludes_zero'] and abs(v['delta']) >= 1.5
                                         for v in contrasts.values()) else 'INDISTINGUISHABLE')
    print(f"\nspread across architectures: {out['spread']['range']:.2f} RMSE "
          f"({out['spread']['min']:.2f} to {out['spread']['max']:.2f})")
    print(f"pre-registered verdict: {out['verdict']}")
    os.makedirs(f'{ROOT}/../results/dsh/e11', exist_ok=True)
    with open(f'{ROOT}/../results/dsh/e11/E11_arch.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('wrote results/dsh/e11/E11_arch.json')


if __name__ == '__main__':
    main()
