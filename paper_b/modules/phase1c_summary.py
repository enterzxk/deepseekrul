"""Summarize Phase 1c queue logs into a 3-seed table.

Usage: python -m paper_b.modules.phase1c_summary --log-dir experiments/phase1c/logs
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def last_block(text):
    parts = re.split(r'^=== RUN .* ===$', text, flags=re.M)
    return parts[-1] if parts else text


def parse_rul(text):
    b = last_block(text)
    out = {}
    m = re.search(r'clean R0 ([\d.]+) R3c ([\d.]+) cost ([+\-]?[\d.]+)', b)
    if m:
        out['R0'], out['R3c'], out['cost'] = map(float, m.groups())
    for arm in ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6_hybrid', 'R7_filter']:
        m = re.search(rf'^{arm} RMSE ([\d.]+)', b, re.M)
        if m:
            out[arm] = float(m.group(1))
    for arm in ['R2', 'R3', 'R4', 'R5', 'R6_hybrid', 'R7_filter']:
        m = re.search(rf'^{arm} recovery (n/a|[\-\d.]+)', b, re.M)
        if m:
            if m.group(1) == 'n/a':
                out[arm + '_rec'] = None
                out['no_harm'] = True
            else:
                out[arm + '_rec'] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='experiments/phase1c/logs')
    ap.add_argument('--out', default='experiments/phase1c/summary.md')
    a = ap.parse_args()
    root = Path(a.log_dir)
    groups = defaultdict(lambda: defaultdict(list))
    eol = {}
    for p in sorted(root.glob('*.log')):
        name = p.name
        text = p.read_text(errors='replace')
        if name.startswith('eol_'):
            b = last_block(text)
            m = re.search(r'EOL_cycle_flag_rate=([\d.]+).*?EOL_any_window=([\d.]+)', b, re.S)
            if m:
                eol[p.stem] = (float(m.group(1)), float(m.group(2)))
            continue
        m = re.match(r'rul_(\w+?)_(\w+)_m(\d+)_s(\d+)\.log', name)
        if not m:
            continue
        subset, fault, mag, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        d = parse_rul(text)
        if 'R3_rec' not in d:
            continue
        key = (subset, fault, mag)
        if d.get('no_harm'):
            groups[key]['no_harm'].append(1)
            if 'cost' in d:
                groups[key]['cost'].append(d['cost'])
            groups[key]['rmse'].append((
                d.get('R0', np.nan), d.get('R1', np.nan), d.get('R3', np.nan),
                d.get('R5', np.nan), d.get('R7_filter', np.nan)))
            continue
        for arm in ['R3', 'R5', 'R6_hybrid', 'R7_filter']:
            v = d.get(arm + '_rec')
            if v is not None:
                groups[key][arm].append(v)
        if 'cost' in d:
            groups[key]['cost'].append(d['cost'])
    lines = ['# Phase 1c summary', '',
             '| subset | fault | mag | n | R3 mean (min–max) | R5 mean (min–max) | R6 mean | R7 mean | clean cost | note |',
             '|---|---|---:|---:|---|---|---|---|---|---|']
    for key in sorted(groups):
        s, f, m = key
        g = groups[key]
        def stat(arm):
            v = np.asarray(g.get(arm, []), float)
            if len(v) == 0:
                return '—'
            return f'{v.mean():.3f} ({v.min():.3f}–{v.max():.3f})'
        note = 'no_harm %d' % len(g['no_harm']) if g.get('no_harm') else ''
        lines.append(f'| {s} | {f} | {m} | {len(g.get("R3", []))} | '
                     f'{stat("R3")} | {stat("R5")} | {stat("R6_hybrid")} | '
                     f'{stat("R7_filter")} | {stat("cost")} | {note} |')
    lines += ['', '## EOL FPR', '', '| subset | per-cycle | any-window |', '|---|---:|---:|']
    for k in sorted(eol):
        lines.append(f'| {k} | {eol[k][0]:.4f} | {eol[k][1]:.4f} |')
    lines += ['', '## no_harm cells (absolute RMSE)', '',
              '| subset | fault | mag | n | R0 | R1 | R3 | R5 | R7 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in sorted(groups):
        g = groups[key]
        if not g.get('rmse'):
            continue
        arr = np.asarray(g['rmse'], float)
        vals = [np.nanmean(arr[:, i]) for i in range(arr.shape[1])]
        lines.append(f'| {key[0]} | {key[1]} | {key[2]} | {len(arr)} | '
                     + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
