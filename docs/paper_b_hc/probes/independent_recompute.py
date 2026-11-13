"""Independent recomputation (manager's audit task).

Deliberately does NOT import aggregate_metrics.py and does NOT read any derived
column from SEED_RESULTS.csv. Everything here comes from metrics.json and
preds.csv read fresh from disk.

Reports
  1. sd(B0) over the six FD001@80 cells            (sample sd, ddof=1)
  2. MAD(M2)/MAD(B0) over the same six seeds       (median absolute deviation)
  3. B2 - B3 at seed s101                          (best checkpoint)
  4. best_val vs val_engine_mean_rmse              (the invariant)
  5. my engine-mean RMSE recomputed from preds.csv vs the reported value
"""
import glob
import json
import os
import statistics

import numpy as np

ROOTS = ['/home/user/sprint24/experiments/hc/_gate',
         '/home/user/sprint24/experiments/hc/d80',
         '/home/user/sprint24/experiments/hc/d',
         '/home/user/sprint24/experiments/hc/d80diag',
         '/home/user/sprint24/experiments/hc/d40diag',
         '/home/user/sprint24/experiments/hc/d80b160']
SEEDS6 = (101, 202, 303, 404, 505, 606)
CAP = 125.0


def load_cells():
    cells = {}
    for root in ROOTS:
        for mf in glob.glob(os.path.join(root, '*', 'metrics.json')):
            d = os.path.dirname(mf)
            try:
                m = json.load(open(mf))
            except Exception:
                continue
            key = (m.get('arm'), m.get('subset'), m.get('seed'))
            ep = int((m.get('budget') or {}).get('epochs') or 0)
            rec = dict(dir=d, root=root, ep=ep, m=m,
                       val=m.get('val_engine_mean_rmse'),
                       best=m.get('best_val_engine_mean_rmse'))
            # prefer the _gate copy when a cell exists in several roots
            if key not in cells or (root.endswith('_gate') and not cells[key]['root'].endswith('_gate')):
                cells[key] = rec
    return cells


def engine_mean_rmse_from_preds(path):
    """Independent implementation: capped label, clipped prediction, per-engine RMSE, mean."""
    with open(path) as f:
        head = f.readline().strip().split(',')
        idx = {name: i for i, name in enumerate(head)}
        need = ['engine_id', 'y_pred', 'y_cap']
        for k in need:
            if k not in idx:
                return None, 'missing column %s (have %s)' % (k, head[:8])
        per = {}
        for line in f:
            parts = line.rstrip('\n').split(',')
            try:
                e = parts[idx['engine_id']]
                p = float(parts[idx['y_pred']])
                y = float(parts[idx['y_cap']])
            except (ValueError, IndexError):
                continue
            p = min(CAP, max(0.0, p))
            per.setdefault(e, []).append((p, y))
    rms = []
    for e, pairs in per.items():
        arr = np.asarray(pairs, dtype=np.float64)
        rms.append(float(np.sqrt(np.mean((arr[:, 0] - arr[:, 1]) ** 2))))
    return float(np.mean(rms)), '%d engines' % len(rms)


cells = load_cells()
print('cells indexed: %d' % len(cells))

b0 = {s: cells.get(('B0', 'FD001', s)) for s in SEEDS6}
m2 = {s: cells.get(('M2', 'FD001', s)) for s in SEEDS6}
print('B0 FD001@80 present:', {s: (b0[s]['ep'] if b0[s] else None) for s in SEEDS6})
print('M2 FD001@80 present:', {s: (m2[s]['ep'] if m2[s] else None) for s in SEEDS6})

vals_b0 = [b0[s]['best'] for s in SEEDS6 if b0[s] and b0[s]['best'] is not None]
vals_m2 = [m2[s]['best'] for s in SEEDS6 if m2[s] and m2[s]['best'] is not None]

print('')
print('== 1. sd(B0) ==')
print('n=%d values=%s' % (len(vals_b0), ['%.4f' % v for v in vals_b0]))
if len(vals_b0) > 1:
    print('mean=%.6f  sd(ddof=1)=%.6f  sd(ddof=0)=%.6f'
          % (statistics.mean(vals_b0), statistics.stdev(vals_b0),
             statistics.pstdev(vals_b0)))

print('')
print('== 2. MAD ratio ==')
if len(vals_b0) > 1 and len(vals_m2) > 1:
    def mad(x):
        med = statistics.median(x)
        return statistics.median([abs(v - med) for v in x])
    mb0, mm2 = mad(vals_b0), mad(vals_m2)
    print('MAD(B0)=%.6f  MAD(M2)=%.6f  ratio=%.6f' % (mb0, mm2, mm2 / mb0))

print('')
print('== 3. B2 - B3 at s101 ==')
b2 = cells.get(('B2', 'FD001', 101))
b3 = cells.get(('B3', 'FD001', 101))
for tag, c in (('B2', b2), ('B3', b3)):
    print('%s: %s' % (tag, ('%s  best=%.4f  ep=%d' % (c['dir'].split('/')[-1], c['best'], c['ep']))
                      if c else 'MISSING'))
if b2 and b3:
    print('B2 - B3 = %.6f' % (b2['best'] - b3['best']))

print('')
print('== 4. invariant: best_val vs val_engine_mean_rmse ==')
bad = [(k, c['best'], c['val']) for k, c in cells.items()
       if c['best'] is not None and c['val'] is not None and abs(c['best'] - c['val']) > 0]
print('cells=%d  violations=%d' % (len(cells), len(bad)))
for k, a, b in bad[:5]:
    print('  VIOLATION', k, a, b)

print('')
print('== 5. recompute engine-mean RMSE from preds.csv ==')
worst = (0.0, None)
checked = 0
for s in SEEDS6:
    for tag, c in (('B0', b0[s]), ('M2', m2[s])):
        if not c:
            continue
        p = os.path.join(c['dir'], 'preds.csv')
        if not os.path.exists(p):
            continue
        mine, note = engine_mean_rmse_from_preds(p)
        if mine is None:
            print('  %s s%d: %s' % (tag, s, note))
            continue
        d = abs(mine - c['val'])
        checked += 1
        if d > worst[0]:
            worst = (d, (tag, s))
        print('  %s s%d: mine=%.6f reported=%.6f  |d|=%.3e  (%s)'
              % (tag, s, mine, c['val'], d, note))
print('checked=%d  worst |mine - reported| = %.3e at %s' % (checked, worst[0], worst[1]))
