"""Summarize Phase 4 R3 vs R5b/R5m logs."""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='experiments/phase4/logs')
    a = ap.parse_args()
    g = defaultdict(lambda: defaultdict(list))
    for p in sorted(Path(a.log_dir).glob('*.log')):
        m = re.match(r'(FD\d{3})_(\w+)_onset([\d.]+)_s(\d+)\.log', p.name)
        if not m:
            continue
        text = p.read_text(errors='replace')
        vals = {}
        for arm in ('R3', 'R5', 'R5b', 'R5m'):
            mm = re.search(rf'{arm} [\d.]+ rec ([\-\d.]+|n/a)', text)
            if mm and mm.group(1) != 'n/a':
                vals[arm] = float(mm.group(1))
        if 'R5m' not in vals:
            continue
        key = (m.group(1), m.group(2), float(m.group(3)))
        for k, v in vals.items():
            g[key][k].append(v)
    print('| subset | fault | onset | n | R3 | R5 | R5b | R5m | R3-max(R5b,R5m) |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for k in sorted(g):
        v = g[k]
        def s(arm):
            x = np.asarray(v.get(arm, []), float)
            return f'{x.mean():.3f}' if len(x) else '—'
        r3 = np.mean(v.get('R3', [np.nan]))
        mx = np.nanmax([np.mean(v.get('R5b', [np.nan])), np.mean(v.get('R5m', [np.nan]))])
        print(f'| {k[0]} | {k[1]} | {k[2]} | {len(v.get("R3", []))} | {s("R3")} | '
              f'{s("R5")} | {s("R5b")} | {s("R5m")} | {r3 - mx:+.3f} |')


if __name__ == '__main__':
    main()
