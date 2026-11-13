"""How sensitive are the module verdicts to the EMA smoothing constant?

ema_min carries an unregistered free parameter. alpha in {0.1,0.2,0.3,0.5} moves
the B2-vs-B3 difference by 0.22, which is the same order as several module
effects. This script recomputes each matched-seed difference at every alpha and
reports whether the SIGN is stable, because a verdict that flips sign with alpha
is not a verdict.

Reads history.json directly; no pipeline import, no SEED_RESULTS.csv.
"""
import glob
import json
import os

ROOTS = ['/home/user/sprint24/experiments/hc/_gate',
         '/home/user/sprint24/experiments/hc/d80',
         '/home/user/sprint24/experiments/hc/d',
         '/home/user/sprint24/experiments/hc/d80diag',
         '/home/user/sprint24/experiments/hc/d40diag']
ALPHAS = (0.1, 0.2, 0.3, 0.5)


def load():
    idx = {}
    for root in ROOTS:
        for d in glob.glob(os.path.join(root, '*', '')):
            d = d.rstrip('/')
            mf = os.path.join(d, 'metrics.json')
            hf = os.path.join(d, 'history.json')
            if not (os.path.exists(mf) and os.path.exists(hf)):
                continue
            try:
                m = json.load(open(mf))
            except Exception:
                continue
            if m.get('subset') != 'FD001':
                continue
            ep = int((m.get('budget') or {}).get('epochs') or 0)
            key = (m.get('arm'), m.get('seed'), ep)
            if key in idx and '_gate' not in root:
                continue
            idx[key] = d
    return idx


def metrics(d):
    h = json.load(open(os.path.join(d, 'history.json')))
    vals = [r['val_engine_mean_rmse'] for r in h]
    out = dict(best=min(vals), tail10=sum(vals[-10:]) / 10.0, n=len(vals))
    for a in ALPHAS:
        e, mn = vals[0], vals[0]
        for v in vals[1:]:
            e = a * v + (1 - a) * e
            mn = min(mn, e)
        out['ema%.1f' % a] = mn
    return out


idx = load()
print('indexed FD001 cells: %d' % len(idx))
for k in sorted(idx):
    print('  %s' % (k,))

PAIRS = [('M2', 'B0'), ('M3B', 'B3'), ('M6', 'B3'), ('B1', 'B0')]
print('')
for arm, base in PAIRS:
    seeds = sorted({k[1] for k in idx if k[0] == arm and k[2] == 80} &
                   {k[1] for k in idx if k[0] == base and k[2] == 80})
    if not seeds:
        print('%s vs %s: no matched seeds at 80ep' % (arm, base))
        continue
    rows = []
    for s in seeds:
        a = metrics(idx[(arm, s, 80)])
        b = metrics(idx[(base, s, 80)])
        rows.append((s, a, b))
    print('%s vs %s  seeds=%s  n=%d' % (arm, base, seeds, len(seeds)))
    for metric in ('best', 'tail10') + tuple('ema%.1f' % a for a in ALPHAS):
        diffs = [r[1][metric] - r[2][metric] for r in rows]
        mean = sum(diffs) / len(diffs)
        signs = {d > 0 for d in diffs}
        stable = len(signs) == 1
        print('   %-7s mean diff %+8.4f   per-seed %s   %s'
              % (metric, mean, ' '.join('%+.3f' % d for d in diffs),
                 'sign-stable' if stable else 'MIXED SIGNS'))
