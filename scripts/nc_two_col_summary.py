#!/usr/bin/env python3
"""Paired two-column summary of the DS02 dev LOEO grid.

Reads experiments/ncmapss/runs_dev.csv and prints, for the two evaluation points
(inner_val = inner-selection epoch*, fixed_passes:200 = fixed 200-pass endpoint):

  1. per-rung mean / sd / n, both columns side by side
  2. per-fold table (mean over seeds), both columns
  3. paired N2 - N3 difference on the same (fold, seed) cell: the MS/SA switch
  4. seed spread inside each cell, next to the rung-to-rung difference
  5. held-out lifetime vs training-mean lifetime, as a ranking confound check

Nothing here selects a column: both are printed at equal weight.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RULES = ('inner_val', 'fixed_passes:200')
RUNGS = ('N0', 'N1', 'N2', 'N3')


def load(path):
    rows = [r for r in csv.DictReader(open(path, encoding='utf-8'))
            if r.get('status') == 'ok' and r.get('rmse_raw_unit')]
    for r in rows:
        r['rmse'] = float(r['rmse_raw_unit'])
        r['seed'] = r['seed'] or 'NA'
    return rows


def fmt(xs, nd=2):
    if not xs:
        return '     -'
    if len(xs) == 1:
        return f'{xs[0]:{nd + 4}.{nd}f}'
    return f'{st.mean(xs):{nd + 4}.{nd}f}±{st.stdev(xs):.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ledger', default='experiments/ncmapss/runs_dev.csv')
    ap.add_argument('--out-root', default='experiments/ncmapss')
    a = ap.parse_args()
    rows = load(a.ledger)
    print(f'rows={len(rows)} from {a.ledger}')
    if not rows:
        return

    # ---- 1. per rung, both columns -------------------------------------
    print('\n=== per rung: RMSE (raw RUL, flight level) ===')
    print(f'{"rung":5s} {"rule":18s} {"mean±sd":>14s} {"n":>4s}')
    for rung in RUNGS:
        for rule in RULES:
            xs = [r['rmse'] for r in rows if r['rung'] == rung and r['selection_rule'] == rule]
            if xs:
                print(f'{rung:5s} {rule:18s} {fmt(xs):>14s} {len(xs):4d}')

    # ---- 2. per fold, mean over seeds ----------------------------------
    print('\n=== per fold (mean over seeds) ===')
    cells = defaultdict(list)
    for r in rows:
        cells[(r['rung'], r['selection_rule'], r['fold'])].append(r['rmse'])
    folds = sorted({r['fold'] for r in rows})
    head = f'{"fold":10s}'
    for rung in RUNGS:
        head += f' {rung + "_in":>11s} {rung + "_fx":>11s}'
    print(head)
    for fold in folds:
        line = f'{fold:10s}'
        for rung in RUNGS:
            for rule in RULES:
                xs = cells.get((rung, rule, fold), [])
                line += f' {fmt(xs, 2):>11s}'
        print(line)

    # ---- 3. paired N2 - N3 (same fold, same seed) ----------------------
    print('\n=== paired N2 - N3 on the same (fold, seed): MS/SA switch effect ===')
    key = {}
    for r in rows:
        key[(r['rung'], r['fold'], r['seed'], r['selection_rule'])] = r['rmse']
    for rule in RULES:
        diffs = []
        for (rung, fold, seed, ru), v in key.items():
            if rung != 'N2' or ru != rule:
                continue
            o = key.get(('N3', fold, seed, rule))
            if o is not None:
                diffs.append((fold, seed, v - o))
        if diffs:
            xs = [d for _, _, d in diffs]
            print(f'{rule:18s} n={len(xs):2d}  mean={st.mean(xs):+7.2f}  '
                  f'sd={st.stdev(xs) if len(xs) > 1 else float("nan"):5.2f}  '
                  f'N3 worse in {sum(d < 0 for d in xs)}/{len(xs)}')
            for fold, seed, d in sorted(diffs):
                print(f'   {fold:10s} s{seed:4s}  N2-N3 = {d:+7.2f}')

    # ---- 4. seed spread vs rung difference ----------------------------
    print('\n=== seed spread inside a cell vs difference between rungs ===')
    per_cell = defaultdict(list)
    for r in rows:
        per_cell[(r['rung'], r['selection_rule'], r['fold'])].append(r['rmse'])
    for rule in RULES:
        sds = [st.stdev(v) for (rung, ru, _), v in per_cell.items()
               if ru == rule and len(v) > 1]
        if sds:
            print(f'{rule:18s} within-cell seed sd: mean={st.mean(sds):.2f} '
                  f'max={max(sds):.2f} (n={len(sds)} cells)')
        m2 = [r['rmse'] for r in rows if r['rung'] == 'N2' and r['selection_rule'] == rule]
        m3 = [r['rmse'] for r in rows if r['rung'] == 'N3' and r['selection_rule'] == rule]
        if m2 and m3:
            print(f'{"":18s} N2-N3 mean gap = {st.mean(m2) - st.mean(m3):+.2f}')

    # ---- 5. lifetime distance -----------------------------------------
    print('\n=== held-out lifetime vs training-mean lifetime ===')
    lt = {}
    for p in Path(a.out_root).glob('*_ev*/unit_lifetime.json'):
        d = json.loads(p.read_text())
        lt[p.parent.name] = {int(k): float(v) for k, v in d.items()}
    seen = set()
    for r in rows:
        fold = r['fold']
        if fold in seen:
            continue
        seen.add(fold)
        # any cell of this fold carries the same lifetime table
        cand = next((v for k, v in lt.items() if k.startswith(f'{r["rung"]}_{fold}')), None)
        if not cand:
            continue
        held = int(r['held_out_engine'])
        train = [v for u, v in cand.items() if u != held]
        print(f'{fold:10s} held_out={held:2d} life={cand.get(held, float("nan")):5.1f}  '
              f'train_mean={st.mean(train):5.1f}  |diff|={abs(cand.get(held, 0) - st.mean(train)):5.1f}')


if __name__ == '__main__':
    main()
