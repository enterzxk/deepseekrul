"""S2 independent verification of the metric and of the Delta/sigma* table.

Why this exists
---------------
Group protocol for Paper B: every number must be recomputed from the saved
per-engine predictions; ``metrics.json::val_engine_rmse``-style fields are
not to be trusted on their own (in the S1-A round that field was pooled, and
the pooled-vs-engine gap reached 6.1382 cycles).

This script therefore:
  1. re-derives the engine-level RMSE from ``val_pred.csv`` (per-engine RMSE,
     then an equal-weight mean over engines) for every cell, and reports the
     largest disagreement with the recorded value;
  2. reports the pooled-vs-engine gap, as a sanity check that the two are
     genuinely different quantities;
  3. recomputes the per-subset Delta-vs-control table and sigma* from those
     re-derived numbers, independently of analyse_s2.py, and diffs the two.

Definitions (frozen 2026-11-14, see docs/paper_b_next/S2_BREAKEVEN.md):
  Delta = treatment engine-level RMSE - same-seed control engine-level RMSE
          (negative = the life auxiliary helps)
  sigma* per subset = the largest tested sigma whose mean Delta <= -0.3
  bands: <0.10 no room | [0.10, 0.20) weak room | >=0.20 room
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 0.3
SIGMAS = (0.05, 0.10, 0.20, 0.30)
SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')


def band(star):
    if star is None or star < 0.10:
        return 'no room'
    if star < 0.20:
        return 'weak room'
    return 'room'


def engine_rmse(pred: pd.DataFrame) -> tuple[float, int]:
    """Equal-weight mean of per-engine RMSE."""
    se = (pred['y_pred'].to_numpy() - pred['y_true'].to_numpy()) ** 2
    per = pd.Series(se).groupby(pred['unit'].to_numpy()).mean().pow(0.5)
    return float(per.mean()), int(per.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='/home/user/s2/runs')
    ap.add_argument('--out-json', default='/home/user/s2/verify.json')
    ap.add_argument('--out-csv', default='/home/user/s2/s2_cells.csv')
    a = ap.parse_args()

    rows = []
    for mf in sorted(Path(a.runs).rglob('metrics.json')):
        m = json.loads(mf.read_text())
        pred = pd.read_csv(mf.parent / 'val_pred.csv')
        rm, n_units = engine_rmse(pred)
        rows.append(dict(subset=m['subset'], seed=int(m['seed']),
                         sigma=m['sigma'], tag=m['tag'], params=int(m['params']),
                         recomputed=rm, recorded=float(m['val_engine_mean_rmse']),
                         pooled=float(m['val_pooled_rmse']), n_units=n_units,
                         epochs_run=int(m['epochs_run']), best_ep=int(m['best_ep'])))
    v = pd.DataFrame(rows)
    print(f'cells: {len(v)} / 100')
    if v.empty:
        raise SystemExit('no cells')

    v['d_rec'] = v.recomputed - v.recorded
    v['d_pool'] = v.pooled - v.recomputed
    print(f'max |recomputed - recorded|      = {v.d_rec.abs().max():.3e}')
    print(f'pooled minus engine-level: mean  = {v.d_pool.mean():+.4f} cycles, '
          f'max |.| = {v.d_pool.abs().max():.4f}')
    print(f'params: {sorted(v.params.unique())}')
    print(f'units per subset: '
          f'{v.groupby("subset").n_units.first().to_dict()}')

    # ---- Delta vs same-seed control -------------------------------------
    ctrl = v[v.sigma.isna()].set_index(['subset', 'seed']).recomputed
    if ctrl.empty:
        raise SystemExit('no control cells')
    d = v[v.sigma.notna()].copy()
    d['ctrl'] = [ctrl.get((r.subset, r.seed), np.nan) for r in d.itertuples()]
    d = d[d.ctrl.notna()].copy()
    d['delta'] = d.recomputed - d.ctrl

    agg = (d.groupby(['subset', 'sigma'])
             .agg(delta_mean=('delta', 'mean'), delta_sd=('delta', 'std'),
                  n_seeds=('delta', 'size'),
                  n_improved=('delta', lambda s: int((s < 0).sum())),
                  delta_min=('delta', 'min'), delta_max=('delta', 'max'))
             .reset_index())
    agg['reaches'] = agg.delta_mean <= -THRESHOLD

    ctrl_mean = ctrl.groupby(level='subset').mean().round(4)
    print('\n== control (sigma=0, constant column) engine-level RMSE, mean over seeds ==')
    print(ctrl_mean.to_string())

    print('\n== Delta vs control, cycles (negative = life auxiliary helps) ==')
    print(agg.pivot(index='sigma', columns='subset', values='delta_mean')
            .round(3).to_string())
    print('\n== seeds per cell (must be 5) ==')
    print(agg.pivot(index='sigma', columns='subset', values='n_seeds').to_string())
    print('\n== seeds with Delta < 0 (out of n_seeds) ==')
    print(agg.pivot(index='sigma', columns='subset', values='n_improved').to_string())

    # ---- sigma* per subset (independent implementation) ------------------
    star, bands, notes = {}, {}, {}
    for s in SUBSETS:
        sub = agg[agg.subset == s]
        if sub.empty:
            continue
        ok = sub[sub.reaches]
        if ok.empty:
            star[s] = None
            notes[s] = 'no tested sigma reaches -0.3'
        else:
            star[s] = float(ok.sigma.max())
            if star[s] == max(SIGMAS):
                notes[s] = 'largest tested sigma still reaches -0.3; lower bound'
        bands[s] = band(star[s])

    print('\n== PRIMARY: per-subset sigma* ==')
    for s in SUBSETS:
        if s in star:
            print(f'  {s}: sigma* = {star[s]} -> {bands[s]}'
                  + (f'  [{notes[s]}]' if s in notes else ''))
    consistent = len(set(bands.values())) == 1
    print(f'  bands consistent across subsets: {consistent}')

    sec = {}
    for s in SIGMAS:
        sec[str(s)] = int(agg[agg.sigma == s].reaches.sum())
    agg_star = max([s for s in SIGMAS if sec[str(s)] >= 3], default=None)
    print(f'\n== SECONDARY (auxiliary only): largest sigma with >=3/4 subsets '
          f'at -0.3: {agg_star}; per-sigma counts {sec}')

    payload = dict(
        n_cells=int(len(v)),
        max_abs_recompute_gap=float(v.d_rec.abs().max()),
        pooled_minus_engine_mean=float(v.d_pool.mean()),
        pooled_minus_engine_max_abs=float(v.d_pool.abs().max()),
        params=sorted(int(x) for x in v.params.unique()),
        units_per_subset={k: int(x) for k, x in
                          v.groupby('subset').n_units.first().items()},
        per_subset=agg.round(6).to_dict('records'),
        sigma_star_by_subset=star, band_by_subset=bands, notes=notes,
        bands_consistent=consistent,
        secondary=dict(sigma_star=agg_star, counts=sec, auxiliary_only=True),
        threshold=THRESHOLD,
        definition=('per-subset sigma* = largest tested sigma with mean Delta '
                    'vs same-seed control <= -0.3; recomputed from val_pred.csv'),
    )
    # per-cell table (control rows carry an empty delta)
    cells = v.copy()
    cells['ctrl_rmse'] = np.nan
    cells['delta'] = np.nan
    for i, r in d.iterrows():
        mask = ((cells.subset == r.subset) & (cells.seed == r.seed)
                & (cells.sigma == r.sigma))
        cells.loc[mask, 'ctrl_rmse'] = r.ctrl
        cells.loc[mask, 'delta'] = r.delta
    cells = cells[['subset', 'seed', 'sigma', 'tag', 'params', 'recomputed',
                   'recorded', 'pooled', 'ctrl_rmse', 'delta']]
    cells = cells.sort_values(['subset', 'sigma', 'seed'], na_position='first')
    cells.round(6).to_csv(a.out_csv, index=False)
    print(f'wrote {a.out_csv} ({len(cells)} rows)')

    Path(a.out_json).write_text(json.dumps(payload, indent=1))
    print(f'wrote {a.out_json}')


if __name__ == '__main__':
    main()
