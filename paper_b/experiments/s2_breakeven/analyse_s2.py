"""S2 analysis: ΔRMSE against sigma, and the breakeven precision sigma*.

Reads /home/user/s2/runs/<subset>/s<seed>/<tag>/metrics.json and reports, per
subset and per seed, the engine-level validation RMSE relative to the no-aux
control. sigma* is the largest sigma whose mean ΔRMSE still drops by at least
0.3 cycles against the control.

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


def tag(sigma):
    return 'ctrl' if sigma is None else f's{int(round(sigma * 100)):03d}'


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
    agg['helps_by_0p3'] = agg.delta_mean <= -THRESHOLD
    agg.to_csv(a.out_csv, index=False)

    print('\n== delta vs control (negative = auxiliary helps), cycles ==')
    piv = agg.pivot(index='sigma', columns='subset', values='delta_mean').round(3)
    print(piv.to_string())
    print('\n== seeds per cell ==')
    print(agg.pivot(index='sigma', columns='subset', values='n_seeds').to_string())

    # sigma*: largest sigma with mean delta <= -0.3 in at least 3 of 4 subsets
    star = None
    detail = {}
    for s in sorted(d['sigma'].unique()):
        hits = int((agg[(agg.sigma == s)].delta_mean <= -THRESHOLD).sum())
        detail[str(s)] = hits
        if hits >= 3:
            star = max(star or 0.0, float(s))
    verdict = ('no room (sigma* < 0.10)' if star is None or star < 0.10
               else 'weak room (0.10 <= sigma* < 0.20)' if star < 0.20
               else 'room (sigma* >= 0.20)')

    payload = dict(threshold=THRESHOLD, rule='mean delta <= -0.3 in >=3 of 4 subsets',
                   sigma_star=star, subsets_passing_by_sigma=detail, verdict=verdict,
                   per_subset=agg.round(4).to_dict('records'),
                   n_runs=int(len(df)))
    Path(a.out_json).write_text(json.dumps(payload, indent=1))
    print(f'\nsigma* = {star}  -> {verdict}')
    print(f'subsets passing per sigma: {detail}')


if __name__ == '__main__':
    main()
