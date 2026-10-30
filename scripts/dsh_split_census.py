import json, glob, hashlib
from collections import defaultdict
import numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob('experiments/*/*/FD002/*/metrics.json')):
    m = json.load(open(f))
    vu = m.get('val_units') or []
    if not vu: continue
    h = hashlib.md5(','.join(map(str, sorted(vu))).encode()).hexdigest()[:8]
    parts = f.split('/')
    rows.append({'root': parts[1], 'cell': parts[2], 'valset': h, 'rmse': m.get('matched_cutoff_rmse'), 'seed': m.get('model_seed')})
df = pd.DataFrame(rows).dropna(subset=['rmse'])
print('runs:', len(df), '| distinct validation sets:', df.valset.nunique())
print(df.groupby('valset').agg(n=('rmse','size'), roots=('root', lambda s: ','.join(sorted(set(s))))).to_string())
multi = df.groupby('cell').valset.nunique()
cells = multi[multi >= 2].index.tolist()
print()
print('cells on >=2 validation sets:', cells)
sd = []
for c in cells:
    g = df[df.cell == c].groupby('valset').rmse.mean()
    if len(g) >= 2:
        sd.append((c, len(g), float(g.std(ddof=1)), float(g.max()-g.min())))
out = pd.DataFrame(sd, columns=['cell','n_valsets','sd','range']).sort_values('sd', ascending=False)
print(out.round(2).to_string(index=False))
if len(out):
    print()
    print('median across-val-set SD:', round(float(out.sd.median()), 2), '| median range:', round(float(out['range'].median()), 2))
