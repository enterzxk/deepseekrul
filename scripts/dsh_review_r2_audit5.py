import glob, json, os, time, statistics
from collections import Counter
rows = []
for mf in glob.glob('experiments/*/*/*/*/metrics.json'):
    try: d = json.load(open(mf))
    except Exception: continue
    if mf.split('/')[1] not in ('order03','split8','split9','levelb'): continue
    rows.append((mf, os.stat(mf).st_mtime, d))
buckets = Counter(int(t // 300) for _, t, _ in rows)
print('5-min completion buckets with >=6 runs:')
for b, v in sorted(buckets.items()):
    if v >= 6:
        print(f'  {time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(b*300))} -> {v} runs')
print()
print('max per 5-min bucket:', max(buckets.values()))
print('total runs in these roots:', len(rows))
ep = [d.get('epochs_ran') or 0 for _,_,d in rows]
print(f'epochs_ran: min={min(ep)} median={int(statistics.median(ep))} max={max(ep)}; >=30: {sum(1 for e in ep if e>=30)}')
