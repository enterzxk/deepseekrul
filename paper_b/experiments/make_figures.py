"""Publication-style figures from all_runs.csv."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CFG = ['base', 'ms', 'sa', 'hi', 'mssa', 'mshi', 'sahi', 'full']
LABEL = {'base': 'Base', 'ms': 'MS', 'sa': 'SA', 'hi': 'HI', 'mssa': 'MS+SA',
         'mshi': 'MS+HI', 'sahi': 'SA+HI', 'full': 'FULL'}
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def grouped_bar(df, tags, labels, title, out, ylabel='Validation RMSE'):
    subsets = sorted(df.subset.unique())
    x = np.arange(len(subsets)); w = 0.8 / max(len(tags), 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=300)
    for i, (tag, lab) in enumerate(zip(tags, labels)):
        m = [df[(df.tag == tag) & (df.subset == s)].val_rmse.mean() for s in subsets]
        e = [df[(df.tag == tag) & (df.subset == s)].val_rmse.std() for s in subsets]
        ax.bar(x + (i - (len(tags) - 1) / 2) * w, m, w, yerr=e, capsize=1.5,
               label=lab, color=COLORS[i % 10], edgecolor='black', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(subsets)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(ncol=4, fontsize=7, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='paper_b/results')
    ap.add_argument('--out', default='paper_b/results/figures')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(a.results) / 'all_runs.csv')
    tags = [f'A8_{c}_L70' for c in CFG if f'A8_{c}_L70' in set(df.tag)]
    if tags:
        grouped_bar(df, tags, [LABEL[t.split('_')[1]] for t in tags],
                    'DSPT internal ablation (validation, 3 seeds)',
                    out / 'fig_ablation8.pdf')
        # relative to base
        base = df[df.tag == 'A8_base_L70'].groupby('subset').val_rmse.mean()
        fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=300)
        subsets = sorted(df.subset.unique()); x = np.arange(len(subsets))
        for i, tag in enumerate(tags[1:]):
            m = [100 * (df[(df.tag == tag) & (df.subset == s)].val_rmse.mean() / base[s] - 1)
                 for s in subsets]
            ax.plot(x, m, marker='o', ms=3, lw=1, label=LABEL[tag.split('_')[1]],
                    color=COLORS[(i + 1) % 10])
        ax.axhline(0, color='black', lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(subsets)
        ax.set_ylabel('Validation RMSE change vs base (%)')
        ax.legend(ncol=4, fontsize=7, frameon=False); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / 'fig_ablation8_relative.pdf'); plt.close(fig)
        print('wrote', out / 'fig_ablation8.pdf')
    for prefix, base_tag, name in (('MSM_', 'A8_base_L70', 'ms'),
                                   ('SAM_', 'A8_sa_L70', 'sa'),
                                   ('HIM_', 'A8_hi_L70', 'hi')):
        tags = [t for t in sorted(df.tag.unique()) if t.startswith(prefix)]
        if not tags:
            continue
        grouped_bar(df, [base_tag] + tags, ['ref'] + [t.replace(prefix, '') for t in tags],
                    f'{name.upper()} mechanism (validation)', out / f'fig_{name}_mech.pdf')


if __name__ == '__main__':
    main()
