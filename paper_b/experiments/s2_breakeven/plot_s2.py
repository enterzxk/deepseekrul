"""S2 figure: per-subset Delta-RMSE vs sigma, from verify.json (or the CSV).

Reads the per-subset table produced by verify_s2_metric.py and draws one line
per subset, with the pre-registered -0.3 cycle threshold as a dashed rule.
Error bars are the across-seed standard deviation of the paired Delta.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ('FD001', 'FD002', 'FD003', 'FD004')
COLORS = {'FD001': '#1f77b4', 'FD002': '#d62728',
          'FD003': '#2ca02c', 'FD004': '#9467bd'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='/home/user/s2/verify.json')
    ap.add_argument('--out', default='/home/user/s2/s2_delta_sigma.png')
    a = ap.parse_args()

    payload = json.loads(Path(a.json).read_text())
    df = pd.DataFrame(payload['per_subset'])

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    for s in ORDER:
        d = df[df.subset == s].sort_values('sigma')
        ax.errorbar(d.sigma, d.delta_mean, yerr=d.delta_sd, marker='o', ms=4,
                    lw=1.4, capsize=2.5, color=COLORS[s], label=s)
    ax.axhline(-0.3, ls='--', lw=1.0, color='0.35')
    ax.text(0.048, -0.36, r'$-0.3$ cycles (pre-registered)', va='top',
            ha='left', fontsize=7.5, color='0.35')
    ax.axhline(0.0, lw=0.8, color='0.8')
    ax.set_xlabel(r'relative noise on total life, $\sigma$')
    ax.set_ylabel(r'$\Delta$RMSE vs no-aux control (cycles)')
    ax.set_title('S2: how precise must the total-life auxiliary be?', fontsize=10)
    ax.set_xscale('log')
    ax.set_xticks(sorted(df.sigma.unique()))
    ax.set_xticklabels([f'{s:g}' for s in sorted(df.sigma.unique())])
    ax.minorticks_off()
    ax.set_xlim(0.043, 0.36)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc='lower right')
    ax.grid(alpha=0.25, lw=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
