"""2x2x2 factorial analysis: main effects, interactions, paired bootstrap, selection.

Factors:
  MS: 'full' (multi-scale k=3/5/7 + projection) vs 'single_proj' (k=3 + projection)
  SA: 'basic' (cross-sensor attention + learned pooling) vs 'mean' (mean pooling)
  HI: lambda=0.5 vs 0 (head always present)
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path('paper_b/results/raw')
SUBS = ['FD001', 'FD002', 'FD003', 'FD004']
FACT = {
    ('single_proj', 'mean', 0): 'F_single_proj_mean_hi0',
    ('single_proj', 'mean', 1): 'F_single_proj_mean_hi1',
    ('single_proj', 'basic', 0): 'F_single_proj_basic_hi0',
    ('single_proj', 'basic', 1): 'F_single_proj_basic_hi1',
    ('full', 'mean', 0): 'F_full_mean_hi0',
    ('full', 'mean', 1): 'F_full_mean_hi1',
    ('full', 'basic', 0): 'F_full_basic_hi0',
    ('full', 'basic', 1): 'F_full_basic_hi1',
}
CTRL = {'C_linear_mean_hi0': 'linear only (no conv)',
        'C_single_static_hi0': 'static sensor weights'}


def load():
    val, test, params, preds = {}, {}, {}, {}
    for tag in list(FACT.values()) + list(CTRL):
        for s in SUBS:
            d = RAW / tag / s / 's101'
            val[(tag, s)] = json.loads((d / 'metrics.json').read_text())['val_engine_rmse'] if (d / 'metrics.json').exists() else np.nan
            if (d / 'test_metrics.json').exists():
                t = json.loads((d / 'test_metrics.json').read_text())
                test[(tag, s)] = t['rmse']; params[tag] = t['params']
            else:
                test[(tag, s)] = np.nan
            vp = d / 'val_pred.csv'
            if vp.exists():
                preds[(tag, s)] = pd.read_csv(vp)
    return val, test, params, preds


def cell_table(val, key='val'):
    rows = []
    for (ms, sa, hi), tag in FACT.items():
        r = dict(ms=ms, sa=sa, hi=hi, tag=tag)
        for s in SUBS:
            r[s] = val[(tag, s)]
        vals = [r[s] for s in SUBS]
        r['mean'] = float(np.mean(vals)) if all(v == v for v in vals) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def effects(tab, col):
    """Main effects and interactions on a per-dataset column (equal-weight over cells)."""
    y = {(r.ms, r.sa, r.hi): getattr(r, col) for r in tab.itertuples()}
    def avg(fixed):
        vals = [v for k, v in y.items() if all(k[i] == fixed[i] for i in fixed)]
        return np.nanmean(vals)
    out = {}
    out['MS'] = avg({0: 'full'}) - avg({0: 'single_proj'})
    out['SA'] = avg({1: 'basic'}) - avg({1: 'mean'})
    out['HI'] = avg({2: 1}) - avg({2: 0})
    # 2-way interactions
    out['MSxSA'] = (avg({0: 'full', 1: 'basic'}) - avg({0: 'full', 1: 'mean'})
                    - avg({0: 'single_proj', 1: 'basic'}) + avg({0: 'single_proj', 1: 'mean'}))
    out['MSxHI'] = (avg({0: 'full', 2: 1}) - avg({0: 'full', 2: 0})
                    - avg({0: 'single_proj', 2: 1}) + avg({0: 'single_proj', 2: 0}))
    out['SAxHI'] = (avg({1: 'basic', 2: 1}) - avg({1: 'basic', 2: 0})
                    - avg({1: 'mean', 2: 1}) + avg({1: 'mean', 2: 0}))
    out['MSxSAxHI'] = (y[('full', 'basic', 1)] - y[('full', 'basic', 0)] - y[('full', 'mean', 1)]
                       + y[('full', 'mean', 0)] - y[('single_proj', 'basic', 1)]
                       + y[('single_proj', 'basic', 0)] + y[('single_proj', 'mean', 1)]
                       - y[('single_proj', 'mean', 0)])
    return out


def bootstrap_pair(preds, tag_a, tag_b, subset, n_boot=1000, seed=0):
    a = preds.get((tag_a, subset)); b = preds.get((tag_b, subset))
    if a is None or b is None:
        return None
    a = a.copy(); b = b.copy()
    a['e2'] = (a.y_pred - a.y_true) ** 2
    b['e2'] = (b.y_pred - b.y_true) ** 2
    ga = a.groupby('unit').e2.mean(); gb = b.groupby('unit').e2.mean()
    units = ga.index.intersection(gb.index).to_numpy()
    ga, gb = ga.loc[units].to_numpy(), gb.loc[units].to_numpy()
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(units), len(units))
        d.append(np.sqrt(ga[idx].mean()) - np.sqrt(gb[idx].mean()))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    val, test, params, preds = load()
    vtab = cell_table(val)
    ttab = cell_table(test)
    print('== validation RMSE (seed 101) ==')
    print(vtab[['tag', 'mean'] + SUBS].round(3).to_string(index=False))
    print('\n== test RMSE (seed 101, descriptive) ==')
    print(ttab[['tag', 'mean'] + SUBS].round(3).to_string(index=False))
    print('\n== main effects / interactions on validation RMSE ==')
    rows = []
    for s in SUBS + ['mean']:
        e = effects(vtab, s)
        e['subset'] = s
        rows.append(e)
    ef = pd.DataFrame(rows)
    print(ef.round(3).to_string(index=False))
    print('\n== main effects / interactions on test RMSE (descriptive) ==')
    rows = []
    for s in SUBS + ['mean']:
        e = effects(ttab, s)
        e['subset'] = s
        rows.append(e)
    eft = pd.DataFrame(rows)
    print(eft.round(3).to_string(index=False))

    # selection: validation mean RMSE, 1% band, then NASA if available, then params
    cand = vtab.dropna(subset=SUBS).sort_values('mean')
    best = cand.iloc[0]
    band = cand[cand['mean'] <= best['mean'] * 1.01]
    print('\n== selection (validation only) ==')
    print(band[['tag', 'mean']].round(4).to_string(index=False))
    final = band.iloc[0]['tag']
    print('FINAL:', final)

    print('\n== paired bootstrap: best factorial cell vs full (validation) ==')
    for s in SUBS:
        r = bootstrap_pair(preds, final, 'F_full_basic_hi1', s)
        print(s, 'delta RMSE (final - full) = %+.3f [%+.3f, %+.3f]' % r if r else 'missing')

    vtab.to_csv('paper_b/results/factorial_val.csv', index=False)
    ttab.to_csv('paper_b/results/factorial_test.csv', index=False)
    ef.to_csv('paper_b/results/factorial_effects_val.csv', index=False)
    eft.to_csv('paper_b/results/factorial_effects_test.csv', index=False)


if __name__ == '__main__':
    main()
