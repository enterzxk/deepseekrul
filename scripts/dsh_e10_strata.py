"""E10 secondary (exploratory): is the per-engine PE-A difference explained by
anything observable about the engine?

The pooled effect is expected to be near zero once all five folds are in. This
script asks the only question that could still make the contrast interesting:
does the sign of the per-engine difference track the engine's lifespan, its
degradation rate, or its operating-condition cluster? If it does, the honest
statement is a *conditional* one ("PE helps on long-lived engines"), which is a
different claim from the unconditional one and needs independent confirmation.

This is explicitly exploratory. It was not pre-registered, it is not a decision
rule, and nothing here can upgrade the pre-registered verdict. It is reported in
the paper only as a mechanism probe, with the label "exploratory".

Outputs results/dsh/e10/E10_strata.json and prints a table.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/yndx/zxk/bearing-rul-pi'
SUBSET = 'FD002'
SEEDS = [2027, 2028, 2029]
FOLDS = [0, 1, 2, 3, 4]
ARMS = ['A', 'PE']

sys.path.insert(0, f'{ROOT}/src')
sys.path.insert(0, f'{ROOT}/scripts')


def find(arm, seed, fold):
    if fold == 0:
        pat = f'{ROOT}/experiments/e3/{arm}_s{seed}/{SUBSET}/{arm}_s{seed}/matched_cutoff_predictions.csv'
    else:
        pat = f'{ROOT}/experiments/e10/{arm}_s{seed}_f{fold}/{SUBSET}/{arm}_s{seed}/matched_cutoff_predictions.csv'
    hits = glob.glob(pat)
    if len(hits) != 1:
        raise FileNotFoundError(pat)
    return hits[0]


def lifespan_table():
    from dataset import load_cmapss_train
    from pathlib import Path
    raw = load_cmapss_train(Path(f'{ROOT}/data/CMAPSS'), SUBSET)
    life = raw.groupby('unit')['cycle'].max().to_dict()
    # degradation rate proxy: mean absolute first difference of the sensor set,
    # normalised by lifespan; uses the full trajectory only as a *description*
    # of the engine, never as an input to any model
    rate = {}
    for u, g in raw.groupby('unit'):
        cols = [c for c in g.columns if c.startswith('s_')][:14]
        d = g[cols].to_numpy(float)
        rate[u] = float(np.abs(np.diff(d, axis=0)).mean() / max(1, len(g) - 1))
    return life, rate


def main():
    life, rate = lifespan_table()
    rows = []
    for f in FOLDS:
        per = {}
        for arm in ARMS:
            stack = []
            units = None
            for i, s in enumerate(SEEDS):
                d = pd.read_csv(find(arm, s, f)).sort_values('unit_id').reset_index(drop=True)
                if units is None:
                    units = d.unit_id.to_numpy()
                else:
                    assert np.array_equal(units, d.unit_id.to_numpy())
                stack.append(d.signed_error.to_numpy(float) ** 2)
            per[arm] = np.mean(stack, axis=0)          # mean squared error per engine
        for i, u in enumerate(units):
            rows.append(dict(fold=f, unit=int(u), mse_a=float(per['A'][i]),
                             mse_pe=float(per['PE'][i]),
                             life=float(life[u]), rate=float(rate[u])))
    df = pd.DataFrame(rows)
    df['d_mse'] = df.mse_pe - df.mse_a          # negative = PE better
    df['d_rmse_pair'] = np.sqrt(df.mse_pe) - np.sqrt(df.mse_a)

    print(f'{len(df)} engine-fold rows from {df.unit.nunique()} distinct engines')

    def summarise(frame, label):
        n = len(frame)
        if n < 5:
            return None
        mean = float(frame.d_rmse_pair.mean())
        se = float(frame.d_rmse_pair.std(ddof=1) / np.sqrt(n))
        wins = float((frame.d_rmse_pair < 0).mean())
        return dict(label=label, n=n, mean=mean, se=se, ci=[mean - 1.96 * se, mean + 1.96 * se],
                    frac_engines_favouring_PE=wins)

    out = {'note': 'exploratory, not pre-registered, cannot upgrade the E10 verdict',
           'overall': summarise(df, 'all'),
           'by_lifespan_tertile': [], 'by_rate_tertile': [], 'by_fold': []}

    q = df['life'].quantile([1 / 3, 2 / 3]).to_numpy()
    for name, lo, hi in (('short', -np.inf, q[0]), ('mid', q[0], q[1]), ('long', q[1], np.inf)):
        sub = df[(df['life'] > lo) & (df['life'] <= hi)]
        r = summarise(sub, name)
        if r:
            out['by_lifespan_tertile'].append(r)
            print(f"  lifespan {name:5s} n={r['n']:3d} mean {r['mean']:+.3f} "
                  f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  PE wins {r['frac_engines_favouring_PE']:.0%}")

    qr = df.rate.quantile([1 / 3, 2 / 3]).to_numpy()
    for name, lo, hi in (('slow', -np.inf, qr[0]), ('mid', qr[0], qr[1]), ('fast', qr[1], np.inf)):
        sub = df[(df.rate > lo) & (df.rate <= hi)]
        r = summarise(sub, name)
        if r:
            out['by_rate_tertile'].append(r)
            print(f"  rate     {name:5s} n={r['n']:3d} mean {r['mean']:+.3f} "
                  f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  PE wins {r['frac_engines_favouring_PE']:.0%}")

    for f in FOLDS:
        r = summarise(df[df.fold == f], f'fold{f}')
        if r:
            out['by_fold'].append(r)
            print(f"  fold     {f}     n={r['n']:3d} mean {r['mean']:+.3f} "
                  f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  PE wins {r['frac_engines_favouring_PE']:.0%}")

    # correlation between the per-engine difference and engine descriptors
    out['corr'] = {
        'd_rmse_vs_life': float(np.corrcoef(df.d_rmse_pair, df['life'])[0, 1]),
        'd_rmse_vs_rate': float(np.corrcoef(df.d_rmse_pair, df.rate)[0, 1]),
        'spearman_vs_life': float(df.d_rmse_pair.corr(df['life'], method='spearman')),
    }
    print(f"\ncorr(d_rmse_pair, life) = {out['corr']['d_rmse_vs_life']:+.3f}   "
          f"spearman {out['corr']['spearman_vs_life']:+.3f}   "
          f"corr with rate {out['corr']['d_rmse_vs_rate']:+.3f}")

    os.makedirs(f'{ROOT}/results/dsh/e10', exist_ok=True)
    df.to_csv(f'{ROOT}/results/dsh/e10/E10_per_engine.csv', index=False)
    with open(f'{ROOT}/results/dsh/e10/E10_strata.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('wrote E10_strata.json and E10_per_engine.csv')


if __name__ == '__main__':
    main()
