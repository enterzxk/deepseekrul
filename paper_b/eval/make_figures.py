"""Generate paper figures from frozen results (run on lab3090)."""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_b.eval.metrics_v2 import engine_mean_rmse, per_engine_rmse

FIG = Path('figures')
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.size': 9, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.fontset': 'stix',
})
C = plt.cm.tab10.colors
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
FRACS = [0.4, 0.55, 0.7, 0.85, 0.95]


def load_preds():
    out = {}
    for s in SUBSETS:
        p = f'results/ltagv2_p1_{s}.preds.csv'
        df = pd.read_csv(p)
        df['rank'] = df.groupby('engine')['cut'].rank(method='dense').astype(int) - 1
        out[s] = df
    return out


def fig1(preds):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for ax, s in zip(axes.ravel(), SUBSETS):
        df = preds[s]
        for col, lab, c in [('p_k1', 'K1', C[0]), ('pred_ltag', 'LTAG', C[1]),
                            ('pred_calib', 'K1-Calib', C[2]), ('pred_ls', 'LS-ctx', C[3])]:
            ys = [engine_mean_rmse(g.y, g[col], g.engine) for _, g in df.groupby('rank')]
            ax.plot(FRACS, ys, marker='o', ms=3, lw=1.4, label=lab, color=c)
        ax.set_title(s, fontsize=9)
        ax.set_xticks(FRACS)
        ax.set_xticklabels([f'{x:.2f}' for x in FRACS])
        ax.set_ylabel('Engine-mean RMSE')
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes[1]:
        ax.set_xlabel('Cut fraction')
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_cutwise.pdf')
    fig.savefig(FIG / 'fig1_cutwise.png')
    plt.close(fig)


def fig2():
    d = json.loads(Path('results/LTAG_test_eval.json').read_text())
    g = {s: [] for s in SUBSETS}
    for k, v in d.items():
        g[k.split('_')[0]].append(v)
    x = np.arange(len(SUBSETS)); w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    for j, (arm, lab, c) in enumerate([('k1', 'K1', C[0]), ('learned', 'LTAG', C[1]),
                                       ('median', 'Median', C[2])]):
        means = [np.mean([r[arm]['rmse'] for r in g[s]]) for s in SUBSETS]
        ax.bar(x + (j - 1) * w, means, w, label=lab, color=c)
        for i, s in enumerate(SUBSETS):
            vals = [r[arm]['rmse'] for r in g[s]]
            ax.scatter(np.full(len(vals), x[i] + (j - 1) * w), vals, s=6,
                       color='k', alpha=0.5, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(SUBSETS)
    ax.set_ylabel('Test RMSE (one window/engine)')
    ax.legend(frameon=False, ncol=3, loc='upper left')
    ax.grid(axis='y', alpha=0.25, lw=0.4)

    ax = axes[1]
    for i, s in enumerate(SUBSETS):
        deltas = [r['k1']['rmse'] - r['learned']['rmse'] for r in g[s]]
        ax.bar(i, np.mean(deltas), 0.55, color=C[1], alpha=0.8)
        ax.scatter(np.full(len(deltas), i), deltas, s=8, color='k', alpha=0.6, zorder=3)
        ax.text(i, np.mean(deltas) + 0.02, f'{len(deltas)}/{len(deltas)}',
                ha='center', fontsize=7)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(SUBSETS)
    ax.set_ylabel(r'$\Delta$RMSE (K1 $-$ LTAG)')
    ax.grid(axis='y', alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_test.pdf')
    fig.savefig(FIG / 'fig2_test.png')
    plt.close(fig)


def fig3(preds):
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    data, labels, wins = [], [], []
    for s in SUBSETS:
        df = preds[s]
        e_ls = per_engine_rmse(df.y, df.pred_ls, df.engine)
        e_lt = per_engine_rmse(df.y, df.pred_ltag, df.engine)
        keys = sorted(e_ls)
        d = np.array([e_ls[k] - e_lt[k] for k in keys])
        data.append(d); labels.append(s); wins.append(int((d < 0).sum()))
    bp = ax.boxplot(data, labels=labels, widths=0.45, patch_artist=True,
                    medianprops=dict(color='k', lw=1.2), showfliers=False)
    for patch, c in zip(bp['boxes'], C):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(0).normal(0, 0.04, len(d)),
                   d, s=8, color='k', alpha=0.5, zorder=3)
        ax.text(i + 1, ax.get_ylim()[1] * 0.92, f'{wins[i]}/{len(d)}', ha='center', fontsize=8)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_ylabel('ΔRMSE (LS − LTAG)')
    ax.set_xlabel('Subset')
    ax.grid(axis='y', alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_winrate.pdf')
    fig.savefig(FIG / 'fig3_winrate.png')
    plt.close(fig)


if __name__ == '__main__':
    preds = load_preds()
    fig1(preds)
    fig2()
    fig3(preds)
    print('figures written')
