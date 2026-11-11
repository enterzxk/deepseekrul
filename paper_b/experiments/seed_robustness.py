"""Seed-robustness check for the selected factorial cell.

Three cells are compared at seeds 101/202/303 on all four C-MAPSS subsets:

  base       F_single_proj_mean_hi0   single k=3 conv, mean pooling, no HI supervision
  cand       F_single_proj_basic_hi1  single k=3 conv, cross-sensor attention, HI supervision
  full       F_full_basic_hi1         multi-scale k=3/5/7 conv, attention, HI supervision

The selection rule is unchanged (validation only, see FACTORIAL_DESIGN.md); the
official test set is reported descriptively and never used for selection.

Statistics reported here:
  * per (cell, subset): mean and sd of the engine-level RMSE over seeds
  * per cell: mean over the four subsets within each seed, then mean/sd over seeds
  * paired deltas (cand - other) per (subset, seed), summarised per subset
  * engine-level bootstrap CI (1000 resamples, paired by engine) per (subset, seed)

A delta whose CI contains 0 is reported as comparable, not as a win.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path('paper_b/results/raw')
SUBS = ['FD001', 'FD002', 'FD003', 'FD004']
SEEDS = [101, 202, 303]
CELLS = {
    'base': 'F_single_proj_mean_hi0',
    'cand': 'F_single_proj_basic_hi1',
    'full': 'F_full_basic_hi1',
}
LABEL = {
    'base': 'single-scale + mean pooling, no HI',
    'cand': 'single-scale + attention + HI (selected)',
    'full': 'multi-scale + attention + HI (FULL)',
}


def load():
    rows = []
    for name, tag in CELLS.items():
        for s in SUBS:
            for sd in SEEDS:
                d = RAW / tag / s / f's{sd}'
                if not (d / 'metrics.json').exists():
                    continue
                m = json.loads((d / 'metrics.json').read_text())
                t = {}
                if (d / 'test_metrics.json').exists():
                    t = json.loads((d / 'test_metrics.json').read_text())
                rows.append(dict(
                    cell=name, tag=tag, subset=s, seed=sd,
                    val_rmse=m['val_engine_rmse'], val_nasa=m['nasa_mean'],
                    test_rmse=t.get('rmse', np.nan), test_nasa=t.get('nasa_mean', np.nan),
                    params=m.get('params'), best_ep=m.get('best_ep'),
                    elapsed_s=m.get('elapsed_s'),
                ))
    return pd.DataFrame(rows)


def summary(raw, metric):
    """mean/sd over seeds for every (cell, subset)."""
    out = (raw.groupby(['cell', 'subset'])[metric]
              .agg(['mean', 'std', 'count']).reset_index())
    return out


def overall(raw, metric):
    """mean over subsets within each seed, then mean/sd over seeds."""
    per_seed = (raw.dropna(subset=[metric])
                   .groupby(['cell', 'seed'])[metric].mean().reset_index())
    out = (per_seed.groupby('cell')[metric]
                   .agg(['mean', 'std', 'count']).reset_index())
    return out, per_seed


def boot_pair(subset, seed, tag_a, tag_b, n_boot=1000, rng=None):
    """Engine-level paired bootstrap of RMSE(a) - RMSE(b)."""
    rng = rng or np.random.default_rng(0)
    pa = RAW / tag_a / subset / f's{seed}' / 'test_pred.csv'
    pb = RAW / tag_b / subset / f's{seed}' / 'test_pred.csv'
    if not (pa.exists() and pb.exists()):
        return None
    a = pd.read_csv(pa)
    b = pd.read_csv(pb)
    a['e2'] = (a.y_pred - a.y_true) ** 2
    b['e2'] = (b.y_pred - b.y_true) ** 2
    ga = a.groupby('unit').e2.mean()
    gb = b.groupby('unit').e2.mean()
    units = ga.index.intersection(gb.index).to_numpy()
    if len(units) < 5:
        return None
    ga = ga.loc[units].to_numpy()
    gb = gb.loc[units].to_numpy()
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(units), len(units))
        d[i] = np.sqrt(ga[idx].mean()) - np.sqrt(gb[idx].mean())
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def paired(raw, metric, a='cand', b='full'):
    rows = []
    for s in SUBS:
        for sd in SEEDS:
            ra = raw[(raw.cell == a) & (raw.subset == s) & (raw.seed == sd)][metric]
            rb = raw[(raw.cell == b) & (raw.subset == s) & (raw.seed == sd)][metric]
            if ra.empty or rb.empty:
                continue
            delta = float(ra.iloc[0] - rb.iloc[0])
            ci = boot_pair(s, sd, CELLS[a], CELLS[b])
            rows.append(dict(subset=s, seed=sd, delta=delta,
                             ci_lo=ci[1] if ci else np.nan,
                             ci_hi=ci[2] if ci else np.nan))
    df = pd.DataFrame(rows)
    df['ci_excludes_zero'] = (df.ci_lo > 0) | (df.ci_hi < 0)
    return df


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def seed_level(raw, metric, a='cand', b='full'):
    """Aggregate the paired deltas across seeds, per subset.

    The engine bootstrap above answers "is this seed's difference larger than
    engine sampling noise". With only three seeds, the honest cross-seed
    statement is a t-interval on the three seed-level deltas, which is wide by
    construction. Sign consistency is reported next to it because with df=2 a
    t-interval will rarely exclude 0 even when the sign never flips.
    """
    rows = []
    for s in SUBS:
        ds = []
        for sd in SEEDS:
            ra = raw[(raw.cell == a) & (raw.subset == s) & (raw.seed == sd)][metric]
            rb = raw[(raw.cell == b) & (raw.subset == s) & (raw.seed == sd)][metric]
            if ra.empty or rb.empty:
                continue
            ds.append(float(ra.iloc[0] - rb.iloc[0]))
        if not ds:
            continue
        arr = np.asarray(ds, dtype=float)
        n = len(arr)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if n > 1 else float('nan')
        half = T975.get(n, float('nan')) * sd / np.sqrt(n) if n > 1 else float('nan')
        rows.append(dict(
            subset=s, n_seeds=n, mean_delta=mean, sd_delta=sd,
            ci_lo=mean - half, ci_hi=mean + half,
            n_cand_better=int((arr < 0).sum()),
            worst_seed_delta=float(arr.max()),
            best_seed_delta=float(arr.min()),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df['sign_consistent'] = (df.n_cand_better == df.n_seeds) | (df.n_cand_better == 0)
        df['t_ci_excludes_zero'] = (df.ci_lo > 0) | (df.ci_hi < 0)
    return df


def fmt_table(sumtab, metric, cells=('base', 'cand', 'full')):
    head = ['cell'] + SUBS + ['mean_over_subsets']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for c in cells:
        vals = []
        for s in SUBS:
            r = sumtab[(sumtab.cell == c) & (sumtab.subset == s)]
            vals.append('%.3f ± %.3f' % (r['mean'].iloc[0], r['std'].iloc[0]) if len(r) else '—')
        per = [sumtab[(sumtab.cell == c) & (sumtab.subset == s)]['mean'].iloc[0]
               for s in SUBS if len(sumtab[(sumtab.cell == c) & (sumtab.subset == s)])]
        vals.append('%.3f' % np.mean(per) if per else '—')
        lines.append('| %s | ' % LABEL[c] + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def main():
    raw = load()
    raw.to_csv('paper_b/results/seed_robustness_raw.csv', index=False)
    n = raw.groupby('cell').size().to_dict()
    print('loaded runs per cell:', n)
    missing = {c: [f'{s}/s{sd}' for s in SUBS for sd in SEEDS
                   if ((raw.cell == c) & (raw.subset == s) & (raw.seed == sd)).sum() == 0]
               for c in CELLS}
    for c, m in missing.items():
        if m:
            print('MISSING', c, m)

    for metric, title in (('val_rmse', 'validation'), ('test_rmse', 'official test (descriptive)')):
        sumtab = summary(raw, metric)
        sumtab.to_csv(f'paper_b/results/seed_robustness_{metric}_summary.csv', index=False)
        ov, per_seed = overall(raw, metric)
        ov.to_csv(f'paper_b/results/seed_robustness_{metric}_overall.csv', index=False)
        print(f'\n== {title} RMSE, mean ± sd over 3 seeds ==')
        print(fmt_table(sumtab, metric))
        print('\n-- per-seed mean over the four subsets --')
        print(per_seed.pivot(index='cell', columns='seed', values=metric).round(3).to_string())
        print('\n-- mean ± sd over seeds (of the per-seed subset means) --')
        print(ov.set_index('cell').round(3).to_string())

    for b in ('full', 'base'):
        df = paired(raw, 'test_rmse', 'cand', b)
        df.to_csv(f'paper_b/results/seed_robustness_paired_vs_{b}.csv', index=False)
        print(f'\n== paired test RMSE delta: cand - {b} (negative = cand better) ==')
        print(df.round(3).to_string(index=False))
        if len(df):
            print('mean delta %+.3f  sd %.3f  n=%d  cand better in %d/%d pairs  '
                  'CI excludes 0 in %d/%d pairs'
                  % (df.delta.mean(), df.delta.std(), len(df),
                     int((df.delta < 0).sum()), len(df),
                     int(df.ci_excludes_zero.sum()), len(df)))
            by = df.groupby('subset').delta.agg(['mean', 'std', 'count'])
            print('\nper subset:')
            print(by.round(3).to_string())

        sl = seed_level(raw, 'test_rmse', 'cand', b)
        sl.to_csv(f'paper_b/results/seed_robustness_seedlevel_vs_{b}.csv', index=False)
        print(f'\n== cross-seed aggregate: cand - {b} (negative = cand better) ==')
        if len(sl):
            print(sl.round(3).to_string(index=False))
            print('sign consistent in %d/%d subsets; t-CI excludes 0 in %d/%d '
                  '(df = n_seeds - 1, t-based)'
                  % (int(sl.sign_consistent.sum()), len(sl),
                     int(sl.t_ci_excludes_zero.sum()), len(sl)))
        else:
            print('no complete subset yet')


if __name__ == '__main__':
    main()
