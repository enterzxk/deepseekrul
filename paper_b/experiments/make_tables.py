"""Build Paper B tables from paper_b/results/all_runs.csv.

All tables are validation-based unless a test_rmse column is present. The
official test is only used after FREEZE.json exists.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CFG_ORDER = ['base', 'ms', 'sa', 'hi', 'mssa', 'mshi', 'sahi', 'full']


def load(root):
    p = Path(root) / 'all_runs.csv'
    if not p.exists():
        raise SystemExit('run analyze.py first: ' + str(p))
    return pd.read_csv(p)


def paired_table(df, base_tag, tags, metric='val_rmse', subsets=None):
    subsets = subsets or sorted(df.subset.unique())
    rows = []
    for tag in tags:
        for subset in subsets:
            b = df[(df.tag == base_tag) & (df.subset == subset)].set_index('seed')[metric]
            t = df[(df.tag == tag) & (df.subset == subset)].set_index('seed')[metric]
            common = b.index.intersection(t.index)
            if len(common) == 0:
                continue
            d = (t.loc[common] - b.loc[common]).values
            rows.append(dict(tag=tag, subset=subset, n=len(common),
                             base_mean=b.loc[common].mean(), tag_mean=t.loc[common].mean(),
                             delta_pct=100 * (t.loc[common].mean() / b.loc[common].mean() - 1),
                             wins=int((d < 0).sum()), deltas=','.join('%.3f' % x for x in d)))
    return pd.DataFrame(rows)


def overall(df, base_tag, tags, metric='val_rmse'):
    rows = []
    for tag in tags:
        ds = []
        for subset in sorted(df.subset.unique()):
            b = df[(df.tag == base_tag) & (df.subset == subset)].set_index('seed')[metric]
            t = df[(df.tag == tag) & (df.subset == subset)].set_index('seed')[metric]
            common = b.index.intersection(t.index)
            if len(common):
                ds.extend(((t.loc[common] - b.loc[common]) / b.loc[common]).values)
        if not ds:
            continue
        ds = np.asarray(ds)
        n = len(ds)
        se = ds.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append(dict(tag=tag, n_pairs=n, mean_delta_pct=100 * ds.mean(),
                         ci95_low=100 * (ds.mean() - 1.96 * se),
                         ci95_high=100 * (ds.mean() + 1.96 * se),
                         wins=int((ds < 0).sum()), win_rate=float((ds < 0).mean())))
    return pd.DataFrame(rows)


def fmt(df, cols=None, pct=True):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == 'f':
            d[c] = d[c].map(lambda x: ('%.2f' % x) if pd.notna(x) else '')
    if cols:
        d = d[cols]
    return d.to_markdown(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='paper_b/results')
    ap.add_argument('--out', default='paper_b/results/tables')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load(a.results)
    metric = 'val_rmse'

    # 1. eight-config DSPT ablation
    tags = [f'A8_{c}_L70' for c in CFG_ORDER if f'A8_{c}_L70' in set(df.tag)]
    if tags:
        t = paired_table(df, 'A8_base_L70', tags, metric)
        o = overall(df, 'A8_base_L70', tags, metric)
        t.to_csv(out / 'ablation8_by_subset.csv', index=False)
        o.to_csv(out / 'ablation8_overall.csv', index=False)
        print('## DSPT 8-config ablation (validation, relative to base)\n')
        print(fmt(o, ['tag', 'n_pairs', 'mean_delta_pct', 'ci95_low', 'ci95_high', 'wins', 'win_rate']))

    # 2. Vanilla vs DSPT
    vt = [t for t in sorted(df.tag.unique()) if t.startswith('VAN_')]
    if vt:
        print('\n## Vanilla baseline (validation RMSE by subset)\n')
        g = df[df.tag.isin(vt)].groupby(['tag', 'subset'])[metric].agg(['mean', 'std', 'count']).reset_index()
        g.to_csv(out / 'vanilla_by_subset.csv', index=False)
        print(fmt(g))

    # 3. mechanism grids
    for prefix, base in (('MSM_', 'A8_base_L70'), ('SAM_', 'A8_sa_L70'), ('HIM_', 'A8_hi_L70')):
        tags = [t for t in sorted(df.tag.unique()) if t.startswith(prefix)]
        if not tags:
            continue
        o = overall(df, base, tags, metric)
        o.to_csv(out / (prefix.lower().strip('_') + '_overall.csv'), index=False)
        print(f'\n## {prefix} mechanism vs {base}\n')
        print(fmt(o, ['tag', 'n_pairs', 'mean_delta_pct', 'ci95_low', 'ci95_high', 'wins', 'win_rate']))

    print('\nwrote tables to', out)


if __name__ == '__main__':
    main()
