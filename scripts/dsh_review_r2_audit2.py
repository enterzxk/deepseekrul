import glob, json, os, time
from collections import defaultdict, Counter
rows = []
for mf in glob.glob('experiments/*/*/*/*/metrics.json'):
    try: d = json.load(open(mf))
    except Exception: continue
    st = os.stat(mf)
    rows.append((mf.split('/')[1], time.gmtime(st.st_mtime), d))
print(f"{'root':20s} {'n':>3s} {'first (UTC)':22s} {'last (UTC)':22s} epochs[min/med/max]")
for root in sorted({r[0] for r in rows}):
    g = sorted([r for r in rows if r[0]==root], key=lambda x: x[1])
    ep = sorted([(r[2].get('epochs_ran') or 0) for r in g])
    f = time.strftime('%Y-%m-%d %H:%M:%SZ', g[0][1]); l = time.strftime('%Y-%m-%d %H:%M:%SZ', g[-1][1])
    print(f'{root:20s} {len(g):3d} {f:22s} {l:22s} {ep[0]}/{ep[len(ep)//2]}/{ep[-1]}')
print()
print('epochs_ran histogram (all runs):', dict(Counter(sorted([(r[2].get('epochs_ran') or 0) for r in rows]))))
print()
runs_after = [r for r in rows if time.mktime(r[1]) >= time.mktime(time.strptime('2026-10-21 12:00:00','%Y-%m-%d %H:%M:%S'))]
print('runs modified after 2026-10-21T12:00Z:', len(runs_after))
print('paths:'); [print('  ', r[0], time.strftime('%Y-%m-%d %H:%M:%SZ', r[1]), r[2].get('epochs_ran')) for r in sorted(runs_after, key=lambda x: x[1])[:20]]
