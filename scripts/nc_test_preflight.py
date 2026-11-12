#!/usr/bin/env python3
"""Compute the test-phase epoch* table from the dev LOEO ledgers, and check it.

For each subset and each (rung, seed) cell this prints

    epoch*(subset, rung, seed) = round_half_up(mean of the six LOEO epoch* values)

which is the number that must be frozen in the manifest *before* the test engines
are scored. It reads only dev ledgers, so running it does not open the test set.

Also asserts the six-fold count, so a half-written or short ledger fails here
rather than after the test batch has been spent.
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

SUBSETS = ('DS02', 'DS05', 'DS06', 'DS07')
RUNGS = ('N0', 'N1', 'N2', 'N3')
SEEDS = {'N0': ('',), 'N1': (101, 202, 303), 'N2': (101, 202, 303), 'N3': (101, 202, 303)}


def half_up(x):
    """Round half away from zero; Python's round() is banker's rounding."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'experiments/ncmapss')
    out = {}
    bad = []
    for sub in SUBSETS:
        led = root / sub / 'runs_dev.csv'
        if not led.exists():
            bad.append(f'{sub}: no dev ledger at {led}')
            continue
        rows = [r for r in csv.DictReader(open(led, encoding='utf-8'))
                if r['selection_rule'] == 'inner_val' and r['status'] == 'ok']
        out[sub] = {}
        for rung in RUNGS:
            for seed in SEEDS[rung]:
                want = '' if rung == 'N0' else str(seed)
                vals = sorted(int(r['epochs_run']) for r in rows
                              if r['rung'] == rung and r['seed'] == want)
                key = f'{rung}' if rung == 'N0' else f'{rung}_s{seed}'
                if len(vals) != 6:
                    bad.append(f'{sub} {key}: {len(vals)} folds, expected 6 -> {vals}')
                    continue
                mean = st.mean(vals)
                out[sub][key] = {'epoch_star': half_up(mean), 'mean': round(mean, 2),
                                 'folds': vals, 'floor_hits': sum(v == 20 for v in vals)}
    print(json.dumps(out, indent=1))
    print('\n--- summary ---')
    for sub, cells in out.items():
        line = ' '.join(f'{k}:{v["epoch_star"]}({v["folds"]})' for k, v in cells.items())
        print(f'{sub}: {line}')
    n = sum(len(v) for v in out.values())
    print(f'\ncells={n} expected=40')
    if bad:
        print('\nFAILURES:')
        for b in bad:
            print(' ', b)
        raise SystemExit(1)
    print('all cells have exactly six dev folds')


if __name__ == '__main__':
    main()
