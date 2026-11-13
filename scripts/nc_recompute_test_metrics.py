#!/usr/bin/env python3
"""Recompute the 20 affected test-row metrics per engine, from pred.csv.

`metrics()` in run_loeo.py aggregates `_unit` and `seg_*` over the whole error
vector, i.e. pooled. On a dev LOEO fold that is indistinguishable from per-engine
equal weight (one engine per fold), so the defect is only observable on a row with
n_units > 1. This script recomputes both aggregations from pred.csv and reports,
per column, whether the two differ -- which is the check that cannot be vacuous.

It also counts, per column, how many rows have per_engine == pooled by coincidence:
those are the columns a difference-based detector would wrongly pass.

    python scripts/nc_recompute_test_metrics.py [experiments/ncmapss] [--write]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CAP = 125.0
SEG_REL = (1 / 3, 2 / 3)
SEG_ABS = (75.0, 25.0)
SUBSETS = ('DS02', 'DS05', 'DS06', 'DS07')

#: the 20 columns the pooled aggregation touched
UNIT_COLS = ('rmse_raw_unit', 'mae_raw_unit', 'bias_raw_unit', 'nasa_raw_unit',
             'rmse_cap125_unit', 'mae_cap125_unit', 'bias_cap125_unit',
             'nasa_cap125_unit')
SEG_COLS = tuple(f'seg_{seg}_{rule}_{crit}'
                 for seg in ('early', 'mid', 'late')
                 for rule in ('relative', 'absolute')
                 for crit in ('capped', 'raw'))
ALL_COLS = UNIT_COLS + SEG_COLS


def _nasa(e):
    a = np.where(e < 0, 1.0 / 13.0, 1.0 / 10.0)
    return float(np.exp(a * np.abs(e)).sum())


def _rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def _cols(y, p, life):
    """Every metric for one engine, computed on that engine's own rows."""
    e = p - y
    yc = np.minimum(y, CAP)
    ec = p - yc
    out = dict(rmse_raw_unit=_rmse(e), mae_raw_unit=float(np.abs(e).mean()),
               bias_raw_unit=float(e.mean()), nasa_raw_unit=_nasa(e),
               rmse_cap125_unit=_rmse(ec), mae_cap125_unit=float(np.abs(ec).mean()),
               bias_cap125_unit=float(ec.mean()), nasa_cap125_unit=_nasa(ec))
    frac = y / np.maximum(life, 1e-9)
    masks = {
        'early_relative': frac > SEG_REL[1],
        'mid_relative': (frac > SEG_REL[0]) & (frac <= SEG_REL[1]),
        'late_relative': frac <= SEG_REL[0],
        'early_absolute': y >= SEG_ABS[0],
        'mid_absolute': (y < SEG_ABS[0]) & (y >= SEG_ABS[1]),
        'late_absolute': y < SEG_ABS[1],
    }
    for name, m in masks.items():
        for crit, err in (('capped', ec), ('raw', e)):
            out[f'seg_{name}_{crit}'] = _rmse(err[m]) if m.any() else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='experiments/ncmapss')
    ap.add_argument('--write', action='store_true', help='write <subset>/runs_test_perunit.csv')
    a = ap.parse_args()
    root = Path(a.root)

    rows_out = defaultdict(list)
    differ_by_col = defaultdict(int)
    equal_by_col = defaultdict(int)
    finite_equal_by_col = defaultdict(int)
    nan_equal_by_col = defaultdict(int)
    rows_affected = 0
    total_rows = 0
    for sub in SUBSETS:
        led = root / sub / 'runs_test.csv'
        if not led.exists():
            print(f'{sub}: no ledger')
            continue
        for r in csv.DictReader(open(led, encoding='utf-8')):
            if r['status'] != 'ok':
                continue
            total_rows += 1
            df = list(csv.DictReader(open(r['preds_path'], encoding='utf-8')))
            units = sorted({int(x['unit']) for x in df})
            per_unit = defaultdict(list)
            for x in df:
                per_unit[int(x['unit'])].append(x)
            life = json.loads((Path(r['preds_path']).parent / 'unit_lifetime.json')
                              .read_text(encoding='utf-8'))
            life = {int(k): float(v) for k, v in life.items()}
            per_eng = []
            for u in units:
                g = per_unit[u]
                y = np.array([float(x['y_true_raw']) for x in g])
                p = np.array([float(x['y_pred']) for x in g])
                per_eng.append(_cols(y, p, life[u]))
            per_engine = {}
            for c in ALL_COLS:
                vals = np.array([d[c] for d in per_eng], float)
                per_engine[c] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
                per_engine[c + '__coverage'] = int(np.isfinite(vals).sum())

            y_all = np.array([float(x['y_true_raw']) for x in df])
            p_all = np.array([float(x['y_pred']) for x in df])
            life_all = np.array([life[int(x['unit'])] for x in df], float)
            pooled = _cols(y_all, p_all, life_all)

            diff_cols = []
            for c in ALL_COLS:
                led_v = float(r[c]) if r.get(c) not in (None, '') else float('nan')
                if not np.isclose(led_v, per_engine[c], rtol=0, atol=1e-6, equal_nan=True):
                    differ_by_col[c] += 1
                    diff_cols.append(c)
                if np.isclose(per_engine[c], pooled[c], rtol=0, atol=1e-9, equal_nan=True):
                    equal_by_col[c] += 1
                    # an empty segment gives nan on both sides: equal but vacuous,
                    # which is a different failure from "coincidentally equal values"
                    if np.isfinite(per_engine[c]):
                        finite_equal_by_col[c] += 1
                    else:
                        nan_equal_by_col[c] += 1
            if diff_cols:
                rows_affected += 1
            out = dict(r)
            for c in ALL_COLS:
                out[c] = per_engine[c]
                out[c + '__coverage'] = per_engine[c + '__coverage']
            out['n_units'] = len(units)
            out['ledger_differs_in'] = ';'.join(diff_cols)
            rows_out[sub].append(out)

    print(f'rows checked: {total_rows}   rows where the ledger differs from '
          f'per-engine: {rows_affected}')
    print(f'{"column":34s} {"ledger!=per_engine":>18s} {"per_engine==pooled":>19s}')
    for c in ALL_COLS:
        print(f'{c:34s} {differ_by_col[c]:18d} {equal_by_col[c]:19d}')
    traps = [c for c in ALL_COLS if finite_equal_by_col[c] > 0 and differ_by_col[c] > 0]
    print(f'\ncolumns that are pooled-equal on some rows and different on others '
          f'(a difference-based detector would pass those rows): {len(traps)}')
    for c in traps:
        print(f'  {c}: finite-equal on {finite_equal_by_col[c]}/{differ_by_col[c] + equal_by_col[c]} rows')
    vac = [c for c in ALL_COLS if nan_equal_by_col[c] > 0]
    print(f'columns with an empty segment on some rows (nan==nan, vacuous): {len(vac)}')
    for c in vac:
        print(f'  {c}: nan on {nan_equal_by_col[c]}/{total_rows} rows')

    if a.write:
        for sub, part in rows_out.items():
            if not part:
                continue
            p = root / sub / 'runs_test_perunit.csv'
            with p.open('w', newline='', encoding='utf-8') as fh:
                w = csv.DictWriter(fh, fieldnames=list(part[0].keys()))
                w.writeheader()
                w.writerows(part)
            print(f'wrote {p}  ({len(part)} rows)')


if __name__ == '__main__':
    main()
