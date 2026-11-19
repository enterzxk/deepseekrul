"""Per-seed Phase 1c table (R0/R1/R3/R5) for seed-spread inspection."""
import argparse
import re
from pathlib import Path

from paper_b.modules.phase1c_summary import parse_rul


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='experiments/phase1c/logs')
    a = ap.parse_args()
    rows = []
    for p in sorted(Path(a.log_dir).glob('rul_*.log')):
        m = re.match(r'rul_(\w+?)_(\w+)_m(\d+)_s(\d+)\.log', p.name)
        if not m:
            continue
        d = parse_rul(p.read_text(errors='replace'))
        if 'R3_rec' not in d:
            continue
        rows.append((m.group(1), m.group(2), int(m.group(3)), int(m.group(4)),
                     d.get('R0', float('nan')), d.get('R1', float('nan')),
                     d.get('R3_rec', float('nan')), d.get('R5_rec', float('nan'))))
    print('| subset | fault | mag | seed | R0 | R1 | R3 | R5 | R3-R5 |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        print(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[5]:.3f} | '
              f'{r[6]:.3f} | {r[7]:.3f} | {r[6] - r[7]:+.3f} |')


if __name__ == '__main__':
    main()
