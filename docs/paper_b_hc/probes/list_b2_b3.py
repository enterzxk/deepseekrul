import glob
import json
import os

ROOTS = glob.glob('/home/user/sprint24/experiments/hc/*/')
rows = []
for root in ROOTS:
    for mf in glob.glob(os.path.join(root, '*', 'metrics.json')):
        try:
            m = json.load(open(mf))
        except Exception:
            continue
        if m.get('arm') in ('B2', 'B3') and m.get('subset') == 'FD001':
            rows.append((m.get('arm'), m.get('seed'),
                         int((m.get('budget') or {}).get('epochs') or 0),
                         m.get('best_val_engine_mean_rmse'),
                         os.path.dirname(mf).replace('/home/user/sprint24/experiments/hc/', '')))
rows.sort(key=lambda r: (r[0], r[1] or 0, r[2]))
print('%-3s %-5s %-5s %-12s %s' % ('arm', 'seed', 'ep', 'best', 'dir'))
for r in rows:
    print('%-3s %-5s %-5s %-12.4f %s' % (r[0], r[1], r[2], r[3] if r[3] is not None else float('nan'), r[4]))
print('')
for a in ('B2', 'B3'):
    vals = [r[3] for r in rows if r[0] == a and r[3] is not None]
    if vals:
        print('%s: n=%d min=%.4f max=%.4f' % (a, len(vals), min(vals), max(vals)))
