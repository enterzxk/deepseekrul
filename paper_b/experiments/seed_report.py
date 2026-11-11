"""Render the three-seed robustness tables into a report.

Reads the CSVs written by ``paper_b.experiments.seed_robustness`` and emits
``paper_b/reports/SEED_ROBUSTNESS.md``. Everything here is transcription: the
script computes counts and labels (how many subsets keep the sign, how many
bootstrap CIs exclude zero) but writes no interpretive prose, so the report
cannot drift away from the CSV evidence.

Usage:
  python -m paper_b.experiments.seed_report
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path('paper_b/results')
SUBS = ['FD001', 'FD002', 'FD003', 'FD004']
LABEL = {
    'base': 'single-scale + mean pooling, no HI',
    'cand': 'single-scale + attention + HI (selected)',
    'full': 'multi-scale + attention + HI (FULL)',
}
ORDER = ['base', 'cand', 'full']


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    return f'{x:.{nd}f}'


def _pm(mean, sd, nd=3):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return '—'
    if sd is None or (isinstance(sd, float) and np.isnan(sd)):
        return f'{mean:.{nd}f} (n=1)'
    return f'{mean:.{nd}f} ± {sd:.{nd}f}'


def table_seed_means(summary, metric, n_seeds):
    """cell x subset mean over seeds, plus the mean over subsets."""
    head = ['cell', *SUBS, 'mean over subsets']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for cell in ORDER:
        row = [LABEL[cell]]
        per = []
        for s in SUBS:
            r = summary[(summary.cell == cell) & (summary.subset == s)]
            if len(r):
                row.append(_pm(float(r['mean'].iloc[0]), float(r['std'].iloc[0])))
                per.append(float(r['mean'].iloc[0]))
            else:
                row.append('—')
        row.append(_fmt(np.mean(per)) if per else '—')
        lines.append('| ' + ' | '.join(row) + ' |')
    if n_seeds < 2:
        lines.append('')
        lines.append(f'`n=1` per cell: no seed variance is estimated yet.')
    return '\n'.join(lines)


def table_paired(paired):
    if not len(paired):
        return '_no paired runs yet_'
    head = ['subset', 'seed', 'delta', '95% CI', 'CI excludes 0']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for _, r in paired.iterrows():
        ci = f'[{_fmt(r.ci_lo)}, {_fmt(r.ci_hi)}]'
        lines.append('| %s | %d | %s | %s | %s |'
                     % (r.subset, int(r.seed), _fmt(r.delta), ci,
                        'yes' if r.ci_excludes_zero else 'no'))
    return '\n'.join(lines)


def table_seed_level(sl):
    if not len(sl):
        return '_no complete subset yet_'
    head = ['subset', 'seeds', 'mean delta', 'sd', 't 95% CI',
            'cand better', 'worst seed', 'sign kept', 't-CI excl. 0']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for _, r in sl.iterrows():
        lines.append('| %s | %d | %s | %s | [%s, %s] | %d/%d | %s | %s | %s |'
                     % (r.subset, int(r.n_seeds), _fmt(r.mean_delta),
                        _fmt(r.sd_delta), _fmt(r.ci_lo), _fmt(r.ci_hi),
                        int(r.n_cand_better), int(r.n_seeds),
                        _fmt(r.worst_seed_delta),
                        'yes' if r.sign_consistent else 'no',
                        'yes' if r.t_ci_excludes_zero else 'no'))
    return '\n'.join(lines)


def counts_block(raw):
    n = raw.groupby('cell').size().to_dict()
    seeds = sorted(raw.seed.unique().tolist()) if len(raw) else []
    parts = [f'seeds present: {", ".join(str(s) for s in seeds) or "none"}']
    parts.append('runs per cell: ' + ', '.join(f'{c}={n.get(c, 0)}' for c in ORDER))
    return '; '.join(parts)


def ranking_block(raw):
    """Does the validation winner stay the winner under every seed?

    Selection was made on a single seed. If a different cell wins under the
    seed-averaged validation RMSE, the selection rule and the reported model
    disagree, and that has to be visible rather than smoothed over.
    """
    v = raw.dropna(subset=['val_rmse'])
    if not len(v):
        return []
    per_seed = (v.groupby(['cell', 'seed']).val_rmse.mean().reset_index())
    seeds = sorted(per_seed.seed.unique().tolist())
    winners = {}
    for sd in seeds:
        col = per_seed[per_seed.seed == sd].sort_values('val_rmse')
        winners[sd] = col.cell.iloc[0]
    agg = per_seed.groupby('cell').val_rmse.mean().sort_values()
    out = ['## Selection stability (validation RMSE)', '',
           '| seed | winner | mean over subsets |', '|---|---|---|']
    for sd in seeds:
        row = per_seed[per_seed.seed == sd].set_index('cell').val_rmse
        out.append('| %d | %s | %s |' % (sd, winners[sd], _fmt(float(row[winners[sd]]))))
    out += ['| **seed mean** | **%s** | **%s** |' % (agg.index[0], _fmt(float(agg.iloc[0]))), '']
    same = all(w == winners[seeds[0]] for w in winners.values())
    out.append('- Winner is `%s` on every seed: **%s**.' % (winners[seeds[0]], 'yes' if same else 'no'))
    out.append('- Winner under the seed-averaged validation RMSE: `%s`%s.'
               % (agg.index[0], '' if agg.index[0] == winners[seeds[0]]
                  else ' — differs from the single-seed winner'))
    out.append('')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(RES))
    ap.add_argument('--out', default='paper_b/reports/SEED_ROBUSTNESS.md')
    a = ap.parse_args()
    res = Path(a.results)

    raw = pd.read_csv(res / 'seed_robustness_raw.csv')
    n_seeds = raw.seed.nunique() if len(raw) else 0

    out = ['# Seed robustness of the selected factorial cell', '',
           'Generated by `python -m paper_b.experiments.seed_report` from',
           '`paper_b/results/seed_robustness_*.csv`. Do not edit by hand.', '',
           f'Coverage: {counts_block(raw)}.', '',
           'Selection rule is unchanged: validation only. The official test set',
           'is reported descriptively and never used to pick the model.', '']

    for metric, title, note in (
        ('val_rmse', 'Validation RMSE (engine level)',
         'Model selection uses this metric.'),
        ('test_rmse', 'Official test RMSE (descriptive)',
         'Opened after the freeze manifest; not a selection criterion.'),
    ):
        f = res / f'seed_robustness_{metric}_summary.csv'
        if not f.exists():
            continue
        summary = pd.read_csv(f)
        out += [f'## {title}', '', note, '',
                table_seed_means(summary, metric, n_seeds), '']
        if metric == 'val_rmse':
            out += ranking_block(raw)

    for b in ('full', 'base'):
        pf = res / f'seed_robustness_paired_vs_{b}.csv'
        sf = res / f'seed_robustness_seedlevel_vs_{b}.csv'
        if not pf.exists():
            continue
        paired = pd.read_csv(pf)
        out += [f'## Paired test RMSE: selected − {b}',
                '', 'Negative delta means the selected cell has lower RMSE.',
                '', table_paired(paired), '']
        if sf.exists():
            sl = pd.read_csv(sf)
            out += ['Cross-seed aggregate (t interval on the per-seed deltas,',
                    f'df = n_seeds − 1; with three seeds the interval is wide by',
                    'construction, so sign consistency is reported alongside it):',
                    '', table_seed_level(sl), '']
            if len(sl):
                out += [f'- Sign kept in **{int(sl.sign_consistent.sum())}/{len(sl)}** subsets.',
                        f'- t interval excludes 0 in **{int(sl.t_ci_excludes_zero.sum())}/{len(sl)}** subsets.',
                        '']
        if len(paired):
            out += [f'- Engine-bootstrap CIs exclude 0 in **{int(paired.ci_excludes_zero.sum())}/{len(paired)}** '
                    'subset-seed pairs.',
                    f'- Mean delta over pairs: **{paired.delta.mean():+.3f}** '
                    f'(sd {paired.delta.std():.3f}).', '']

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(out), encoding='utf-8')
    print('wrote', a.out, len(out), 'lines')


if __name__ == '__main__':
    main()
