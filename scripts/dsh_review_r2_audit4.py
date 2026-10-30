import glob, json, os, time
from collections import Counter
rows = []
for mf in glob.glob('experiments/*/*/*/*/metrics.json'):
    try: d = json.load(open(mf))
    except Exception: continue
    rows.append((mf, time.gmtime(os.stat(mf).st_mtime), d))
# order03 + split8 + split9 + levelb only
sel = [(p,t,d) for p,t,d in rows if p.split('/')[1] in ('order03','split8','split9','levelb')]
print('runs in order03/split8/split9/levelb:', len(sel))
per_root = Counter(p.split('/')[1] for p,_,_ in sel)
print('per root:', dict(per_root))
e30 = sum(1 for _,_,d in sel if (d.get('epochs_ran') or 0) >= 30)
print('runs with epochs_ran>=30:', e30, '| <30:', len(sel)-e30)
print()
# concurrency: count runs whose mtime falls within the same 5-minute bucket
buckets = Counter(time.strftime('%m-%d %H:%M', time.gmtime(t.tm_hour*0+t.tm_min//5*5*60)) for _,t,_ in sel)
print('max runs sharing a 5-min completion bucket:', max(buckets.values()))
for k,v in sorted(buckets.items()): 
    if v >= 6: print('  bucket', k, '->', v, 'runs')
