"""Independent recompute of the TEST NASA score as a SUM (C-MAPSS original).

`test_metrics.json` stores `pooled_nasa_mean`, not the sum, so
`nasa_sum == pooled_nasa_mean * n` is an identity, not a check. This script
recomputes the per-engine NASA score from `test_predictions.csv` and sums it,
then compares the sum with `pooled_nasa_mean * n` from the metrics file.

NASA score, C-MAPSS definition (d = y_pred - y_true):
    s = exp(-d/13) - 1   if d < 0   (early)
    s = exp( d/10) - 1   if d >= 0  (late)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from _cells import resolve                                    # noqa: E402

ARMS = ('E0', 'E1a', 'E1b', 'E2a')
SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = {17: (101, 202, 303), 29: (42, 43, 101, 202, 303)}


def nasa(d: float) -> float:
    return math.exp(-d / 13.0) - 1.0 if d < 0 else math.exp(d / 10.0) - 1.0


def cell_nasa(cell: Path):
    """(sum of per-engine NASA score, n, mean) for the clean TEST rows.

    The row filter must match `audit_test.py` exactly -- same two predicates, same
    duplicate-engine assertion. A looser filter here would make "NASA recomputed
    from predictions" and "Delta computed from predictions" disagree about which
    rows are the clean TEST set, and the disagreement would be invisible because
    both would still produce a number.
    """
    tot, n = 0.0, 0
    seen = set()
    with open(cell / 'test_predictions.csv', newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('corruption_type') not in ('none', '', None):
                continue
            if row.get('evaluation_protocol') not in ('clean_test', '', None):
                continue
            u = int(row['engine_id'])
            if u in seen:
                raise SystemExit(f'{cell}: duplicate engine_id {u} in clean rows')
            seen.add(u)
            yt, yp = float(row['y_true']), float(row['y_pred_evaluated'])
            tot += nasa(yp - yt)
            n += 1
    return tot, n, tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    root = Path(a.root)

    out, bad = {}, []
    for split in (17, 29):
        seeds = SEEDS[split]
        table = {}
        for arm in ARMS:
            for subset in SUBSETS:
                sums, ns, means = [], [], []
                for seed in seeds:
                    cell = Path(resolve(root, arm, subset, seed, split))
                    tot, n, mean = cell_nasa(cell)
                    sums.append(tot)
                    ns.append(n)
                    means.append(mean)
                    # identity check: mean from metrics file must equal sum/n
                    rec = json.loads((cell / 'test_metrics.json').read_text())
                    rec_mean = rec.get('pooled_nasa_mean')
                    if rec_mean is not None and abs(rec_mean - mean) > 1e-6:
                        bad.append(dict(arm=arm, subset=subset, seed=seed, split=split,
                                        recomputed=mean, recorded=rec_mean))
                    if rec.get('n') != n:
                        bad.append(dict(arm=arm, subset=subset, seed=seed, split=split,
                                        n_recomputed=n, n_recorded=rec.get('n')))
                if len(set(ns)) != 1:
                    raise SystemExit(f'{arm} {subset} split{split}: n differs across seeds {ns}')
                table[(arm, subset)] = (sum(sums) / len(sums), ns[0])
        out[str(split)] = {f'{arm}/{s}': dict(nasa_sum=v[0], n=v[1])
                           for (arm, s), v in table.items()}
        print(f'== split {split}')
        for arm in ARMS:
            row = []
            better = 0
            for subset in SUBSETS:
                v = table[(arm, subset)][0]
                row.append(f'{subset} {v:8.2f}')
                if arm != 'E0' and v < table[('E0', subset)][0]:
                    better += 1
            tag = '' if arm == 'E0' else f'  better {better}/4'
            print(f'{arm:4} ' + '  '.join(row) + tag)

    print(f'identity/consistency problems: {len(bad)}')
    for b in bad[:8]:
        print('   ', b)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(tables=out, problems=bad), indent=1),
                                encoding='utf-8')
        print(f'wrote {a.json}')
    return 0 if not bad else 1


if __name__ == '__main__':
    raise SystemExit(main())
