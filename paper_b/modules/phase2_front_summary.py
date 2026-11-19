"""Summarize M_front logs (frozen M0 + static RHSE-TSH)."""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='experiments/phase2/logs')
    a = ap.parse_args()
    g = defaultdict(list)
    for p in sorted(Path(a.log_dir).glob('mfront_*.log')):
        m = re.match(r'mfront_(FD\d{3})_s(\d+)_(\w+)\.log', p.name)
        if not m:
            continue
        text = p.read_text(errors='replace')
        r3 = re.search(r'R3 [\d.]+ recovery ([\-\d.]+|n/a)', text)
        r5 = re.search(r'R5 [\d.]+ recovery ([\-\d.]+|n/a)', text)
        if not r3:
            continue
        g[(m.group(1), m.group(3))].append((
            float(r3.group(1)) if r3.group(1) != 'n/a' else np.nan,
            float(r5.group(1)) if r5 and r5.group(1) != 'n/a' else np.nan))
    print('| subset | fault | n | R3 mean (min–max) | R5 mean |')
    print('|---|---|---:|---|---:|')
    for k in sorted(g):
        v = np.asarray(g[k], float)
        r3 = v[:, 0]; r5 = v[:, 1]
        print(f'| {k[0]} | {k[1]} | {len(v)} | '
              f'{np.nanmean(r3):.3f} ({np.nanmin(r3):.3f}–{np.nanmax(r3):.3f}) | '
              f'{np.nanmean(r5):.3f} |')


if __name__ == '__main__':
    main()
