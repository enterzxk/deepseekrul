"""Summarize Phase 2 fault comparison logs (M0Aug/M1/M2 vs M0)."""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='experiments/phase2/logs')
    a = ap.parse_args()
    g = defaultdict(lambda: defaultdict(list))
    for p in sorted(Path(a.log_dir).glob('fault_*.log')):
        m = re.match(r'fault_(FD\d{3})_s(\d+)_(\w+)\.log', p.name)
        if not m:
            continue
        text = p.read_text(errors='replace')
        for arm in ('M0Aug', 'M1', 'M2'):
            mm = re.search(rf'{arm} [\d.]+ rec ([\-\d.]+|n/a)', text)
            if mm and mm.group(1) != 'n/a':
                g[(m.group(1), m.group(3))][arm].append(float(mm.group(1)))
    print('| subset | fault | n | M0Aug | M1 | M2 |')
    print('|---|---|---:|---:|---:|---:|')
    for k in sorted(g):
        v = g[k]
        def s(arm):
            x = np.asarray(v.get(arm, []), float)
            return f'{x.mean():.3f} ({x.min():.3f}–{x.max():.3f})' if len(x) else '—'
        print(f'| {k[0]} | {k[1]} | {len(v.get("M1", []))} | {s("M0Aug")} | {s("M1")} | {s("M2")} |')


if __name__ == '__main__':
    main()
