# Provenance (header added 2026-11-13 by zxk; the code below is unchanged apart from the
# ledger path, which moved to DS02_pilot/ when the pilot batch was archived).
#   produced: DS02 six-fold table (fixed: N0 9.01 / N2 10.66 / N3 13.53 / N1 26.32), lifetime-distance check, seed-vs-fold variance
#   output:   stdout only; reads the archived pilot ledger, cited in NCMAPSS_FINAL_RESULTS.md
#   run:      python -m paper_b.ncmapss.probes.final_analysis   (from the repo root)
"""Final DS02 dev analysis: six-fold table, lifetime-distance check, variance split.

Reads experiments/ncmapss/DS02_pilot/runs_dev.csv (paired rows per cell) and prints:
  1. per-rung, per-fold RMSE for both evaluation points, with the held-out
     engine's lifetime and the training-mean lifetime next to it;
  2. the lifetime-distance check: does the ranking track |lifetime - train mean|?
  3. seed vs fold variance against the between-rung difference.
"""
from __future__ import annotations

import argparse
import collections
import csv
import statistics as st

#: The sealed DS02 dev table is the authority: the pilot's working ledger was
#: archived and then lost, and the sealed copy reproduces every published number
#: (checked 2026-11-13: 9.009 / 26.317 / 10.662 / 13.534 at fixed_passes:200).
#: Pass --ledger to run this on a live batch, e.g.
#:   --ledger experiments/ncmapss/DS05/runs_dev.csv
LEDGER = 'docs/paper_b_ncmapss/registry/runs_DS02.csv'
RUNGS = ('N0', 'N1', 'N2', 'N3')
LIFETIME = {2: 74, 5: 88, 10: 81, 16: 62, 18: 70, 20: 65}  # max(Y), the same denominator evaluation.py uses
RULES = ('fixed_passes:200', 'inner_val')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ledger', default=LEDGER,
                    help='ledger to analyse (default: the sealed DS02 dev table)')
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.ledger, encoding='utf-8'))
            if r.get('status') == 'ok']
    print(f'rows={len(rows)} from {args.ledger}')

    cells = collections.defaultdict(dict)          # (rung, rule, fold) -> rmse
    for r in rows:
        v = r.get('rmse_raw_unit')
        if v:
            # `fold` is the label ('LOEO-2'); the engine number is its own column
            cells[(r['rung'], r['selection_rule'], int(r['held_out_engine']))][r['seed']] = float(v)

    folds = sorted({k[2] for k in cells})
    held = {}
    for k in cells:
        held[k[2]] = None

    print('\n=== per-fold RMSE (mean over seeds) ===')
    header = 'rung  rule              ' + ''.join(f'  f{f}(u{LIFETIME.get(f, "?")})' for f in folds)
    print(header)
    table = collections.defaultdict(dict)
    for rung in RUNGS:
        for rule in RULES:
            vals = []
            line = f'{rung:5s} {rule:18s}'
            for f in folds:
                d = cells.get((rung, rule, f), {})
                m = st.mean(d.values()) if d else float('nan')
                vals.append(m)
                table[(rung, rule)][f] = m
                line += f'{m:10.2f}'
            print(line)

    print('\n=== lifetime-distance check ===')
    train_mean = {f: st.mean([LIFETIME[u] for u in LIFETIME if u != f]) for f in folds}
    print('fold  held_life  train_mean  distance')
    for f in folds:
        print(f'{f:5d} {LIFETIME[f]:9d} {train_mean[f]:11.1f} {LIFETIME[f]-train_mean[f]:9.1f}')

    print('\n  rank of N2-N3 gap per fold vs distance (does the gap track lifetime?)')
    for f in folds:
        g = table[('N3', 'fixed_passes:200')][f] - table[('N2', 'fixed_passes:200')][f]
        print(f'  fold {f:2d}  u={LIFETIME[f]:2d}  dist={LIFETIME[f]-train_mean[f]:6.1f}  N3-N2={g:7.2f}')

    print('\n=== variance split (fixed_passes:200) ===')
    for rung in RUNGS:
        per_seed = []
        per_fold = []
        for f in folds:
            d = cells.get((rung, 'fixed_passes:200', f), {})
            if d:
                per_seed.append(st.pstdev(list(d.values())) if len(d) > 1 else 0.0)
                per_fold.append(st.mean(d.values()))
        fold_sd = st.pstdev(per_fold) if len(per_fold) > 1 else 0.0
        seed_sd = st.mean(per_seed) if per_seed else 0.0
        print(f'{rung:4s} seed_sd={seed_sd:6.2f}  fold_sd={fold_sd:6.2f}  '
              f'fold_mean={st.mean(per_fold):7.2f}')

    print('\n=== between-rung differences (fold-paired, fixed_passes:200) ===')
    for a, b in (('N2', 'N3'), ('N0', 'N2'), ('N1', 'N2')):
        diffs = [table[(b, 'fixed_passes:200')][f] - table[(a, 'fixed_passes:200')][f] for f in folds]
        if all(d == d for d in diffs):
            print(f'{b} - {a}: mean={st.mean(diffs):7.2f}  per-fold={[round(d,2) for d in diffs]}  '
                  f'sign_consistent={len({d>0 for d in diffs})==1}')


if __name__ == '__main__':
    main()
