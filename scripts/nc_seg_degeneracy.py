#!/usr/bin/env python3
"""Why do per-engine and pooled agree on some rows but not others?

Hypothesis: the aggregation is vacuous whenever every engine contributes exactly
one flight to the aggregated set -- then "mean over engines" and "mean over
flights" are the same arithmetic mean. That covers the dev case (one engine per
fold) and any segment in which each test engine has exactly one flight.

This counts, per row and per segment, how many flights each engine contributes,
and reports the rows where the minimum is 1 (degenerate) versus more than 1.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

CAP = 125.0
SEG_REL = (1 / 3, 2 / 3)
SEG_ABS = (75.0, 25.0)
SUBSETS = ('DS02', 'DS05', 'DS06', 'DS07')


def main():
    root = Path('experiments/ncmapss')
    degenerate = defaultdict(int)
    nondegenerate = defaultdict(int)
    for sub in SUBSETS:
        led = root / sub / 'runs_test.csv'
        if not led.exists():
            continue
        for r in csv.DictReader(open(led, encoding='utf-8')):
            if r['status'] != 'ok':
                continue
            df = list(csv.DictReader(open(r['preds_path'], encoding='utf-8')))
            life = json.loads((Path(r['preds_path']).parent / 'unit_lifetime.json')
                              .read_text(encoding='utf-8'))
            life = {int(k): float(v) for k, v in life.items()}
            per = defaultdict(list)
            for x in df:
                u = int(x['unit'])
                y = float(x['y_true_raw'])
                frac = y / max(life[u], 1e-9)
                per[('early_relative', u)].append(frac > SEG_REL[1])
                per[('mid_relative', u)].append(SEG_REL[0] < frac <= SEG_REL[1])
                per[('late_relative', u)].append(frac <= SEG_REL[0])
                per[('early_absolute', u)].append(y >= SEG_ABS[0])
                per[('mid_absolute', u)].append(SEG_ABS[1] <= y < SEG_ABS[0])
                per[('late_absolute', u)].append(y < SEG_ABS[1])
            units = sorted({int(x['unit']) for x in df})
            for seg in ('early_relative', 'mid_relative', 'late_relative',
                        'early_absolute', 'mid_absolute', 'late_absolute'):
                counts = [sum(per[(seg, u)]) for u in units]
                if not any(counts):
                    continue
                # degenerate when every engine that has data contributes exactly one
                if min(c for c in counts if c) == 1 and max(counts) == 1:
                    degenerate[seg] += 1
                else:
                    nondegenerate[seg] += 1
    print(f'{"segment":18s} {"degenerate rows":>15s} {"non-degenerate":>14s}')
    for seg in ('early_relative', 'mid_relative', 'late_relative',
                'early_absolute', 'mid_absolute', 'late_absolute'):
        print(f'{seg:18s} {degenerate[seg]:15d} {nondegenerate[seg]:14d}')
    print('\ndegenerate = every engine contributes exactly one flight, so per-engine '
          'equal weight and pooling are the same arithmetic mean')


if __name__ == '__main__':
    main()
