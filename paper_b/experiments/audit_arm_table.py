"""Unified arm table for the DSPT-Base functional audit.

Reads saved val_pred.csv files from one or more result roots (earlier roots win),
computes engine-level validation RMSE, and prints the comparison tables used in
docs/paper_b_baselines/DSPT_BASE_FUNCTIONAL_AUDIT.md.

Usage:
  python -m paper_b.experiments.audit_arm_table \
      --roots paper_b/results/raw_a40,paper_b/results/raw \
      --out paper_b/results/audit_arm_table.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = [
    ('DSPT+3ch+cycle', 'A8_base_L70'),
    ('Van+raw', 'VAN_raw_L70'),
    ('Van+raw+cycle', 'VAN_raw_cycle_L70'),
    ('Van+3ch', 'VAN_dsp_L70'),
    ('Van+3ch+cycle', 'VAN_dsp_cycle_L70'),
]
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']


def engine_level(csv_path):
    d = pd.read_csv(csv_path)
    per = []
    for _, g in d.groupby('unit'):
        e = g.y_pred.to_numpy() - g.y_true.to_numpy()
        per.append(float(np.sqrt(np.mean(e ** 2))))
    per = np.asarray(per)
    return float(per.mean()), float(per.std(ddof=1)), int(len(per))


def collect(roots, tags, a8_csv=None):
    rows, used = [], {}
    a8 = pd.read_csv(a8_csv) if a8_csv and Path(a8_csv).exists() else None
    if a8 is not None:
        a8['seed'] = a8['seed'].str.replace('s', '', regex=False).astype(int)
    for tag in tags:
        for subset in SUBSETS:
            for seed in (101, 202, 303):
                hit = None
                for root in roots:
                    p = Path(root) / tag / subset / f's{seed}' / 'val_pred.csv'
                    if p.exists():
                        mean, sd, n = engine_level(p)
                        hit = dict(tag=tag, subset=subset, seed=seed, n_engines=n,
                                   val_eng=mean, val_eng_sd=sd, source=root)
                        break
                if hit is None and a8 is not None:
                    r = a8[(a8.tag == tag) & (a8.subset == subset) & (a8.seed == seed)]
                    if len(r) == 1:
                        hit = dict(tag=tag, subset=subset, seed=seed, n_engines=np.nan,
                                   val_eng=float(r.val_eng.iloc[0]), val_eng_sd=np.nan,
                                   source=a8_csv)
                if hit is not None:
                    rows.append(hit)
                    used[(tag, subset, seed)] = hit['source']
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', default='paper_b/results/raw_a40,paper_b/results/raw')
    ap.add_argument('--a8-csv', default='paper_b/results/a8_engine_level.csv')
    ap.add_argument('--out', default='paper_b/results/audit_arm_table.csv')
    a = ap.parse_args()
    roots = [r for r in a.roots.split(',') if Path(r).exists()]
    df = collect(roots, [t for _, t in ARMS], a.a8_csv)
    df.to_csv(a.out, index=False)

    print('| arm | ' + ' | '.join(SUBSETS) + ' | mean |')
    print('|---|' + '---|' * (len(SUBSETS) + 1))
    for name, tag in ARMS:
        d = df[df.tag == tag]
        cells = []
        for s in SUBSETS:
            v = d[d.subset == s].val_eng
            cells.append(f'{v.mean():.3f} ± {v.std(ddof=1):.3f}' if len(v) > 1 else 'NA')
        print(f'| {name} | ' + ' | '.join(cells) + f' | {d.val_eng.mean():.3f} |')

    print('\ncoverage:')
    print(df.groupby('tag').size().to_string())
    print('\nsource roots:')
    print(df.groupby(['tag', 'source']).size().to_string())

    # ---- decomposition (mean difference per subset; arms may span machines) ----
    def mean_of(tag, subset):
        return df[(df.tag == tag) & (df.subset == subset)].val_eng.mean()

    def effect(a, b, label):
        out = []
        for s in SUBSETS:
            out.append(f'{mean_of(a, s) - mean_of(b, s):+.3f}')
        print(f'| {label} | ' + ' | '.join(out) + ' |')

    print('\ndecomposition (difference of subset means; negative = first arm better):')
    print('| effect | ' + ' | '.join(SUBSETS) + ' |')
    print('|---|' + '---|' * len(SUBSETS))
    effect('A8_base_L70', 'VAN_dsp_cycle_L70', 'architecture @ 3ch+cycle (DSPT − Van)')
    effect('VAN_dsp_cycle_L70', 'VAN_dsp_L70', 'cycle @ Van+3ch (with − without)')
    effect('VAN_raw_cycle_L70', 'VAN_raw_L70', 'cycle @ Van+raw (with − without)')
    effect('VAN_dsp_L70', 'VAN_raw_L70', '3ch @ Van, no cycle')
    effect('VAN_dsp_cycle_L70', 'VAN_raw_cycle_L70', '3ch @ Van, with cycle')


if __name__ == '__main__':
    main()
