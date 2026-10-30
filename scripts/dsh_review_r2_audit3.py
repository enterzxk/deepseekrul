import glob, json, os, time
from collections import Counter
rows = []
for mf in glob.glob('experiments/*/*/*/*/metrics.json'):
    try: d = json.load(open(mf))
    except Exception: continue
    rows.append((mf.split('/')[1], time.gmtime(os.stat(mf).st_mtime), d, mf))
print(f"{'root':20s} {'n':>3s} {'zero-epoch':>10s} {'first_utc':>20s} {'last_utc':>20s}")
for root in sorted({r[0] for r in rows}):
    g = sorted([r for r in rows if r[0]==root], key=lambda x: x[1])
    z = sum(1 for r in g if (r[2].get('epochs_ran') or 0) == 0)
    f = time.strftime('%m-%d %H:%M', g[0][1]); l = time.strftime('%m-%d %H:%M', g[-1][1])
    print(f'{root:20s} {len(g):3d} {z:10d} {f:>20s} {l:>20s}')
print()
print('zero-epoch runs by root and sample path:')
for r in rows:
    if (r[2].get('epochs_ran') or 0) == 0:
        print('  ', r[0], r[3]); break
print('  total zero-epoch:', sum(1 for r in rows if (r[2].get('epochs_ran') or 0)==0))
