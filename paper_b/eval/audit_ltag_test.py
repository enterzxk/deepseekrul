"""Audit the frozen B0 LTAG test metrics (no test inference)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(path='tmp/claude_pkg/results/LTAG_test_eval.json'):
    d = json.loads(Path(path).read_text())
    g = defaultdict(list)
    for key, v in d.items():
        g[key.split('_')[0]].append((v['k1']['rmse'], v['learned']['rmse'], v['median']['rmse']))
    print('| subset | seeds | K1 | LTAG | Median | delta | wins |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    deltas = []
    for s in sorted(g):
        rows = g[s]
        k1 = np.mean([r[0] for r in rows]); lt = np.mean([r[1] for r in rows])
        md = np.mean([r[2] for r in rows]); dl = k1 - lt
        deltas.append(dl)
        print(f'| {s} | {len(rows)} | {k1:.3f} | {lt:.3f} | {md:.3f} | {dl:+.3f} | '
              f'{sum(r[1] < r[0] for r in rows)}/{len(rows)} |')
    print(f'overall mean delta = {np.mean(deltas):+.3f}')


if __name__ == '__main__':
    main()
