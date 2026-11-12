"""Audit the calibration.json of every calibrated run.

Two things a reviewer needs from these files and cannot get from the RMSE table:

  1. Is the calibration seed-invariant? It must be: the gain is a function of
     (shared init, subset, encoding) only. If it moved with the seed, then the
     seed spread would mix calibration noise into training noise and the
     "seed range = lower bound on training randomness" claim would break.
  2. Is the target ratio actually met? rms_ratio_after must be rho to float
     precision, otherwise the arms are not at a common operating point.

Reads only. Usage: python docs/paper_b_hc/probes/cal_audit.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul')
OUT = ROOT / 'experiments/hc'
PAT = re.compile(r'^HC-C-(A2cal|A3bcal)-(FD00[1-4])-s(\d+)-W2')


def main():
    cells = defaultdict(list)
    for run in sorted(OUT.glob('HC-C-*cal-*')):
        m = PAT.match(run.name)
        f = run / 'calibration.json'
        if not m or not f.exists():
            continue
        rec = json.loads(f.read_text())
        cells[(m.group(1), m.group(2))].append((int(m.group(3)), run.name, rec))

    print('%-7s %-6s %5s %20s %14s %12s'
          % ('arm', 'subset', 'seeds', 'gain', 'ratio_after', 'seed-spread'))
    bad = []
    for (arm, subset), rows in sorted(cells.items()):
        gains = [r['gain'] for _, _, r in rows]
        ratios = [r['rms_ratio_after'] for _, _, r in rows]
        spread = max(gains) - min(gains)
        print('%-7s %-6s %5d %20.10f %14.10f %12.1e'
              % (arm, subset, len(rows), gains[0], max(ratios), spread))
        if spread > 1e-12:
            bad.append('%s %s: gain moves with the seed (spread %.3e)' % (arm, subset, spread))
        for g, rt in zip(gains, ratios):
            if abs(rt - rows[0][2]['rho']) > 1e-6:
                bad.append('%s %s: ratio_after %.9f != rho %.3f' % (arm, subset, rt, rows[0][2]['rho']))

    print('\n%d (arm, subset) groups, %d runs' % (len(cells), sum(len(v) for v in cells.values())))
    if bad:
        print('PROBLEMS:')
        for b in bad:
            print('  ' + b)
    else:
        print('OK: gain is seed-invariant within every (arm, subset); ratio_after == rho')

    print('\ncross-subset gain spread (the reason FD001 cannot be used as a constant):')
    for arm in ('A2cal', 'A3bcal'):
        gs = {s: cells[(arm, s)][0][2]['gain'] for (a, s) in cells if a == arm}
        if gs:
            lo, hi = min(gs.values()), max(gs.values())
            print('  %-7s min %.10f (%s)  max %.10f (%s)  spread %.1f%%'
                  % (arm, lo, min(gs, key=gs.get), hi, max(gs, key=gs.get),
                     100.0 * (hi - lo) / lo))


if __name__ == '__main__':
    main()
