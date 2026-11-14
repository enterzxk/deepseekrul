"""S2 analysis: per-subset ΔRMSE against sigma and per-subset sigma*.

Pre-registered definition (frozen 2026-11-14, not to be changed after results
arrive):

  * PRIMARY: sigma* is defined **per subset** — the largest sigma whose mean
    ΔRMSE (treatment minus no-aux control) is still <= -0.3 cycles. Four
    numbers, one per subset.
  * The band (no room / weak room / room) is applied **per subset**. If the
    subsets land in different bands, that disagreement is reported as such:
    no median, no majority vote.
  * SECONDARY (auxiliary only): "largest sigma where >=3 of 4 subsets reach
    -0.3". Reported for convenience, never as a substitute for the four
    per-subset values.

Usage: python analyse_s2.py [--out-json ...] [--out-csv ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path('/home/user/s2/runs')
SIGMAS = (0.05, 0.10, 0.20, 0.30)
THRESHOLD = 0.3          # cycles, "roughly seed noise" per the specification
SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
BANDS = ((0.10, 'no room'), (0.20, 'weak room'))


def tag(sigma):
    return 'ctrl' if sigma is None else f's{int(round(sigma * 100)):03d}'


def band(star):
    """Map a per-subset sigma* to its pre-registered band."""
    if star is None or star < BANDS[0][0]:
        return 'no room'
    if star < BANDS[1][0]:
        return 'weak room'
    return 'room'


def load():
    rows = []
    for f in sorted(RUNS.rglob('metrics.json')):
        m = json.loads(f.read_text())
        rows.append(dict(subset=m['subset'], seed=m['seed'], sigma=m['sigma'],
                         tag=m['tag'], rmse=m['val_engine_mean_rmse'],
                         nasa=m['nasa_sum'], best_ep=m['best_ep'],
                         epochs=m['epochs_run'], secs=m['elapsed_s']))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-json', default='/home/user/s2/analysis.json')
    ap.add_argument('--out-csv', default='/home/user/s2/delta_by_sigma.csv')
    a = ap.parse_args()

    df = load()
    print(f'runs present: {len(df)} / 100')

    ctrl = df[df.sigma.isna()].set_index(['subset', 'seed']).rmse
    if ctrl.empty:
        raise SystemExit('no control runs yet')

    recs = []
    for _, r in df[df.sigma.notna()].iterrows():
        key = (r.subset, r.seed)
        if key not in ctrl.index:
            continue
        recs.append(dict(subset=r.subset, seed=r.seed, sigma=r.sigma,
                         rmse=r.rmse, ctrl_rmse=float(ctrl.loc[key]),
                         delta=float(r.rmse - ctrl.loc[key])))   # negative = better
    d = pd.DataFrame(recs)
    if d.empty:
        raise SystemExit('no treatment runs yet')

    agg = (d.groupby(['subset', 'sigma'])
             .agg(delta_mean=('delta', 'mean'), delta_sd=('delta', 'std'),
                  n_seeds=('delta', 'size'))
             .reset_index())
    agg['reaches_minus_0p3'] = agg.delta_mean <= -THRESHOLD
    agg.to_csv(a.out_csv, index=False)

    print('\n== delta vs control (negative = auxiliary helps), cycles ==')
    print(agg.pivot(index='sigma', columns='subset', values='delta_mean').round(3).to_string())
    print('\n== seeds per cell ==')
    print(agg.pivot(index='sigma', columns='subset', values='n_seeds').to_string())

    # ---- PRIMARY: per-subset sigma* -------------------------------------
    star_by_subset, band_by_subset, note_by_subset = {}, {}, {}
    for subset in SUBSETS:
        sub = agg[agg.subset == subset]
        if sub.empty:
            continue
        ok = sub[sub.delta_mean <= -THRESHOLD]
        if ok.empty:
            star_by_subset[subset] = None
            note_by_subset[subset] = 'no tested sigma reaches -0.3'
        else:
            star_by_subset[subset] = float(ok.sigma.max())
            if float(ok.sigma.max()) == max(SIGMAS):
                note_by_subset[subset] = ('largest tested sigma still reaches -0.3; '
                                          'sigma* is a lower bound, not a ceiling')
        band_by_subset[subset] = band(star_by_subset[subset])

    bands = list(band_by_subset.values())
    consistent = len(set(bands)) == 1
    print('\n== PRIMARY: per-subset sigma* ==')
    for s in SUBSETS:
        if s in star_by_subset:
            print(f'  {s}: sigma* = {star_by_subset[s]}  -> {band_by_subset[s]}'
                  + (f'  [{note_by_subset[s]}]' if s in note_by_subset else ''))
    print(f'  bands consistent across subsets: {consistent}'
          + ('' if consistent else '  (report the disagreement, do not vote)'))

    # ---- SECONDARY: cross-subset aggregate (auxiliary only) -------------
    detail = {}
    for s in sorted(d['sigma'].unique()):
        detail[str(s)] = int(agg[(agg.sigma == s)].reaches_minus_0p3.sum())
    agg_star = None
    for s in sorted(d['sigma'].unique()):
        if detail[str(s)] >= 3:
            agg_star = max(agg_star or 0.0, float(s))
    print(f'\n== SECONDARY (auxiliary): largest sigma with >=3/4 subsets at -0.3: {agg_star}')
    print(f'   subsets passing per sigma: {detail}')

    payload = dict(
        threshold=THRESHOLD,
        definition=('per-subset sigma* = largest tested sigma whose mean delta vs the '
                    'no-aux control is <= -0.3; bands applied per subset'),
        frozen_on='2026-11-14',
        primary=dict(sigma_star_by_subset=star_by_subset,
                     band_by_subset=band_by_subset,
                     notes=note_by_subset,
                     bands_consistent=consistent),
        secondary_aggregate=dict(rule='largest sigma with >=3/4 subsets at -0.3',
                                 sigma_star=agg_star, subsets_passing=detail,
                                 auxiliary_only=True),
        per_subset=agg.round(4).to_dict('records'),
        n_runs=int(len(df)),
    )
    Path(a.out_json).write_text(json.dumps(payload, indent=1))
    print(f'\nwrote {a.out_json}')


if __name__ == '__main__':
    main()
