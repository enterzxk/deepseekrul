import json, glob, hashlib
from pathlib import Path
cur = hashlib.sha256(Path('src/candidates/model.py').read_bytes()).hexdigest()
print('current model.py sha256:', cur[:16])
rows = []
for f in sorted(glob.glob('experiments/*/*/FD002/*/metrics.json')):
    m = json.load(open(f))
    p = f.split('/')
    rows.append((p[1], p[2], (m.get('code_sha256') or '')[:16], m.get('config', {}).get('gate_mode', '-')))
from collections import Counter
c = Counter((r[2]) for r in rows)
print('distinct code hashes:', len(c))
for h, n in c.most_common():
    print(f'  {h}: {n} runs')
print()
print('gate runs:')
for r in rows:
    if r[1].startswith('H2') or r[1] in ('M4_ckpt','M4_racdr','M4c_noid','M4b_popgate','M5_globalgate'):
        print('  ', r[0], r[1], 'hash=', r[2], 'gate_mode=', r[3])
