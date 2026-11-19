"""Summarize Phase 2 arm runs (M0, M0Aug, ...)."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='experiments/phase2')
    a = ap.parse_args()
    root = Path(a.root)
    rows = []
    for p in sorted(root.glob('phase2_*_s*/metrics.json')):
        m = re.match(r'phase2_(\w+?)_(FD\d{3})_s(\d+)$', p.parent.name)
        if not m:
            continue
        d = json.loads(p.read_text())
        rows.append((m.group(1), m.group(2), int(m.group(3)),
                     d.get('val_engine_mean_rmse'), d.get('val_pooled_rmse'),
                     d.get('best_ep'), d.get('epochs_run')))
    print('| arm | subset | seed | engine_mean | pooled | best_ep | epochs |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for r in rows:
        print(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} | {r[4]:.4f} | {r[5]} | {r[6]} |')
    print('')
    g = defaultdict(list)
    for r in rows:
        g[(r[0], r[1])].append(r[3])
    print('| arm | subset | n | mean | min | max |')
    print('|---|---|---:|---:|---:|---:|')
    for k in sorted(g):
        v = np.asarray(g[k], float)
        print(f'| {k[0]} | {k[1]} | {len(v)} | {v.mean():.4f} | {v.min():.4f} | {v.max():.4f} |')


if __name__ == '__main__':
    main()
