"""Measure the N-CMAPSS DS02 HDF5 file and emit the data dictionary evidence.

This is the runtime half of ``docs/paper_b_ncmapss/DATASET_PROTOCOL.md`` §2: it
checks the file against the documented structure (Data 6(1):5, 2021, Tables
3-8) and records every quantity the protocol depends on.

    python -m paper_b.ncmapss.inspect_ds02 --h5 data/NCMAPSS/N-CMAPSS_DS02-006.h5 \
        --out experiments/ncmapss/ds02_audit.json

Nothing here trains anything and nothing writes to the dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import (A_COLS, DEV_UNITS, T_COLS, TEST_UNITS, W_COLS, X_S_COLS,
                   X_V_COLS, load_raw, verify)


def _summary(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(n=int(a.size), n_nan=int(np.isnan(a).sum()),
                min=float(np.nanmin(a)), max=float(np.nanmax(a)),
                mean=float(np.nanmean(a)), std=float(np.nanstd(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', default='experiments/ncmapss/ds02_audit.json')
    a = ap.parse_args()
    path = Path(a.h5)
    rep = verify(path)

    # ---- documented vs actual column names -----------------------------
    doc = {'W': W_COLS, 'X_s': X_S_COLS, 'X_v': X_V_COLS, 'T': T_COLS, 'A': A_COLS}
    rep['columns_match_docs'] = {}
    for name, expected in doc.items():
        got = rep['arrays'].get(f'{name}_dev', {}).get('cols')
        if got is None:
            continue
        rep['columns_match_docs'][name] = dict(
            match_exact=tuple(got) == tuple(expected),
            match_as_set=set(got) == set(expected),
            missing=sorted(set(expected) - set(got)),
            extra=sorted(set(got) - set(expected)),
        )

    # ---- per-split unit / cycle / label structure ----------------------
    rep['splits'] = {}
    for split, expected_units in (('dev', DEV_UNITS), ('test', TEST_UNITS)):
        d = load_raw(path, split, arrays=('A', 'Y'))
        A, Y = d['A'], d['Y']
        cols = d['A_cols']
        ix = {c: i for i, c in enumerate(cols)}
        unit = A[:, ix['unit']].astype(np.int64)
        cycle = A[:, ix['cycle']].astype(np.int64)
        fc = A[:, ix['Fc']].astype(np.int64)
        hs = A[:, ix['hs']].astype(np.int64)
        units = sorted(int(u) for u in np.unique(unit))
        per_unit = {}
        for u in units:
            m = unit == u
            cyc = cycle[m]
            uq = np.unique(cyc)
            rows_per_cycle = np.array([int((cyc == c).sum()) for c in uq])
            per_unit[int(u)] = dict(
                n_rows=int(m.sum()), n_cycles=int(len(uq)),
                cycle_min=int(uq.min()), cycle_max=int(uq.max()),
                cycle_contiguous=bool(np.array_equal(uq, np.arange(uq.min(), uq.min() + len(uq)))),
                rows_per_cycle_median=float(np.median(rows_per_cycle)),
                rows_per_cycle_min=int(rows_per_cycle.min()),
                rows_per_cycle_max=int(rows_per_cycle.max()),
                fc=sorted(int(x) for x in np.unique(fc[m])),
                hs_values=sorted(int(x) for x in np.unique(hs[m])),
                hs_fault_share=float((hs[m] == 0).mean()) if set(np.unique(hs[m])) <= {0, 1} else None,
                y_min=float(Y[m].min()), y_max=float(Y[m].max()),
                y_at_last_cycle=float(Y[m][cyc == uq.max()].mean()),
                y_at_first_cycle=float(Y[m][cyc == uq.min()].mean()),
            )
        # is Y constant inside a (unit, cycle) cell?
        key = unit.astype(np.int64) * 1_000_000 + cycle
        order = np.argsort(key, kind='stable')
        ks, ys = key[order], Y[order]
        bounds = np.flatnonzero(np.diff(ks)) + 1
        groups = np.split(ys, bounds)
        spread = np.array([float(np.ptp(g)) for g in groups])
        rep['splits'][split] = dict(
            units=units,
            units_match_official=sorted(units) == sorted(expected_units),
            n_rows=int(len(unit)),
            n_units=len(units),
            fc_per_unit={int(u): v['fc'] for u, v in per_unit.items()},
            hs_values=sorted(int(x) for x in np.unique(hs)),
            y=_summary(Y),
            y_within_cycle_max_spread=float(spread.max()),
            y_constant_within_cycle=bool(spread.max() < 1e-9),
            per_unit=per_unit,
        )

    # ---- feature-array sanity -----------------------------------------
    feats = load_raw(path, 'dev', arrays=('W', 'X_s'))
    rep['feature_arrays'] = {k: _summary(feats[k]) for k in ('W', 'X_s')}
    # forbidden arrays are loaded only to record that they exist and are separate
    rep['forbidden_arrays_present'] = {
        k: bool(rep['arrays'].get(f'{k}_dev', {}).get('present'))
        for k in ('X_v', 'T')
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding='utf-8')

    # ---- console summary ----------------------------------------------
    print(f'file {path.name}  {rep["size_bytes"]/1e9:.2f} GB')
    print('keys:', ', '.join(rep['keys']))
    for split in ('dev', 'test'):
        s = rep['splits'][split]
        print(f'[{split}] units={s["units"]} (official={s["units_match_official"]}) '
              f'rows={s["n_rows"]:,}  Y=[{s["y"]["min"]:.1f},{s["y"]["max"]:.1f}] '
              f'Y-per-cycle-constant={s["y_constant_within_cycle"]}')
        for u, v in s['per_unit'].items():
            print(f'   unit {u:>2}: rows={v["n_rows"]:>7,} cycles={v["n_cycles"]:>3} '
                  f'rows/cycle~{v["rows_per_cycle_median"]:.0f} '
                  f'Fc={v["fc"]} hs={v["hs_values"]} '
                  f'Y first/last={v["y_at_first_cycle"]:.1f}/{v["y_at_last_cycle"]:.1f}')
    for name, v in rep['columns_match_docs'].items():
        print(f'columns {name}: exact={v["match_exact"]} set={v["match_as_set"]} '
              f'missing={v["missing"]} extra={v["extra"]}')
    print('wrote', out)


if __name__ == '__main__':
    main()
