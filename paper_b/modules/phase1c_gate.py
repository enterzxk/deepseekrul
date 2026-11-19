"""Evaluate Phase 1c logs against the pre-registered Gate 1.

Usage: python -m paper_b.modules.phase1c_gate --log-dir experiments/phase1c/logs
Exit code 0 = all checks pass; 1 = at least one fail; 2 = incomplete grid.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper_b.modules.phase1c_summary import last_block, parse_rul


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='experiments/phase1c/logs')
    a = ap.parse_args()
    root = Path(a.log_dir)
    cells = defaultdict(lambda: defaultdict(list))
    eol = []
    for p in sorted(root.glob('*.log')):
        text = p.read_text(errors='replace')
        if p.name.startswith('eol_'):
            b = last_block(text)
            m = re.search(r'EOL_cycle_flag_rate=([\d.]+)', b)
            if m:
                eol.append(float(m.group(1)))
            continue
        m = re.match(r'rul_(\w+?)_(\w+)_m(\d+)_s(\d+)\.log', p.name)
        if not m:
            continue
        d = parse_rul(text)
        key = (m.group(1), m.group(2), int(m.group(3)))
        if d.get('no_harm'):
            cells[key]['no_harm'].append(1)
            continue
        for arm in ['R3', 'R5', 'R6_hybrid']:
            v = d.get(arm + '_rec')
            if v is not None:
                cells[key][arm].append(v)
        if 'cost' in d:
            cells[key]['cost'].append(d['cost'])

    print('# Phase 1c Gate 1', '')
    clean_max = max((np.mean(v['cost']) for v in cells.values() if v['cost']), default=float('nan'))
    eol_max = max(eol, default=float('nan'))
    print(f'- clean cost max: {clean_max:.4f} (gate <=0.15)')
    print(f'- EOL per-sensor-cycle max: {eol_max:.4f} (gate <=0.02)')
    print('')
    print('| subset | fault | mag | n | R3 mean | R5 mean | R3-R5 | R3>=0.65 |')
    print('|---|---|---:|---:|---:|---:|---:|---|')
    all_ok = True
    for key in sorted(cells):
        s, f, m = key
        r3 = np.asarray(cells[key].get('R3', []), float)
        r5 = np.asarray(cells[key].get('R5', []), float)
        if cells[key].get('no_harm') and len(r3) == 0:
            print(f'| {s} | {f} | {m} | n/a | n/a | n/a | n/a | PASS(no_harm) |')
            continue
        r3m = r3.mean() if len(r3) else float('nan')
        r5m = r5.mean() if len(r5) else float('nan')
        gap = r3m - r5m
        if f == 'noise':
            print(f'| {s} | {f} | {m} | {len(r3)} | {r3m:.3f} | {r5m:.3f} | '
                  f'{gap:+.3f} | noise (separate) |')
            continue
        ok = (len(r3) >= 3) and r3m >= 0.65
        all_ok &= ok
        print(f'| {s} | {f} | {m} | {len(r3)} | {r3m:.3f} | {r5m:.3f} | '
              f'{gap:+.3f} | {"PASS" if ok else "FAIL"} |')
    n_cells = len(cells)
    print('')
    print(f'- cells: {n_cells} (expected 20: 2 subsets x (3 faults x 3 mags + stuck x 1))')
    if n_cells < 20:
        print('- INCOMPLETE: queue still running or cells missing')
        raise SystemExit(2)
    if clean_max > 0.15 or eol_max > 0.02 or not all_ok:
        print('- GATE 1: FAIL')
        raise SystemExit(1)
    print('- GATE 1: PASS')


if __name__ == '__main__':
    main()
