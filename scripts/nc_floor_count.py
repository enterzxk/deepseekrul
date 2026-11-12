#!/usr/bin/env python3
"""Count how many dev LOEO folds stop at the 20-pass floor, per subset and rung.

Both denominators that have appeared in the documents are printed, so the
"20 of 24" vs "14 of 24" discrepancy can be settled mechanically instead of by
argument: floor hits among N0 folds (the claim under dispute) and floor hits among
all inner_val rows of DS02 (where the 20 likely came from).
"""
import csv
import sys
from pathlib import Path

SUBSETS = ('DS02', 'DS05', 'DS06', 'DS07')
FLOOR = 20


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'experiments/ncmapss')
    per_subset = {}
    for sub in SUBSETS:
        p = root / sub / 'runs_dev.csv'
        rows = [r for r in csv.DictReader(open(p, encoding='utf-8'))
                if r['selection_rule'] == 'inner_val' and r['status'] == 'ok']
        n0 = [int(r['epochs_run']) for r in rows if r['rung'] == 'N0']
        hits = [v for v in n0 if v == FLOOR]
        per_subset[sub] = (len(hits), len(n0), n0)
        print(f'{sub}: N0 folds at floor {len(hits)}/{len(n0)}  values={n0}')

    tot_h = sum(v[0] for v in per_subset.values())
    tot_n = sum(v[1] for v in per_subset.values())
    print(f'\nN0 across four subsets: {tot_h}/{tot_n} folds at the {FLOOR}-pass floor')

    ds02 = root / 'DS02' / 'runs_dev.csv'
    rows = [r for r in csv.DictReader(open(ds02, encoding='utf-8'))
            if r['selection_rule'] == 'inner_val' and r['status'] == 'ok']
    at_floor = [r for r in rows if int(r['epochs_run']) == FLOOR]
    print(f'DS02 all inner_val rows: {len(at_floor)}/{len(rows)} at the floor '
          f'(by rung: ' +
          ', '.join(f'{r}={sum(1 for x in at_floor if x["rung"] == r)}'
                    for r in ('N0', 'N1', 'N2', 'N3')) + ')')


if __name__ == '__main__':
    main()
