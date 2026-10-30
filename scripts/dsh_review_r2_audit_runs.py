import glob, json, os, time
rows = []
for mf in glob.glob('experiments/*/*/*/*/metrics.json'):
    root = mf.split('/')[1]
    try:
        d = json.load(open(mf))
    except Exception:
        continue
    st = os.stat(mf)
    rows.append({
        'root': root, 'path': mf,
        'mtime_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(st.st_mtime)),
        'epochs_ran': d.get('epochs_ran'), 'seed': d.get('model_seed'),
        'split_seed': d.get('split_seed'), 'params': d.get('params'),
        'code_sha': (d.get('provenance') or {}).get('code_sha256', '')[:12],
        'has_prov': bool(d.get('provenance')),
    })
rows.sort(key=lambda r: r['mtime_utc'])
print('total metrics.json:', len(rows))
from collections import Counter
print('by root:', dict(Counter(r['root'] for r in rows)))
print()
print('roots with provenance block:', dict(Counter(r['root'] for r in rows if r['has_prov'])))
print()
print('earliest 3:'); [print('  ', r['mtime_utc'], r['path']) for r in rows[:3]]
print('latest 6:');   [print('  ', r['mtime_utc'], r['path'], 'epochs=', r['epochs_ran']) for r in rows[-6:]]
