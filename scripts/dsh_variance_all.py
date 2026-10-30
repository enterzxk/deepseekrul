import json, glob, hashlib
import numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob('experiments/*/*/FD002/*/metrics.json')):
    m = json.load(open(f)); vu = m.get('val_units') or []
    if not vu: continue
    h = hashlib.md5(','.join(map(str, sorted(vu))).encode()).hexdigest()[:8]
    p = f.split('/')
    rows.append({'cell': p[2], 'valset': h, 'seed': m.get('model_seed'),
                 'rmse': m.get('matched_cutoff_rmse')})
df = pd.DataFrame(rows).dropna()
y = df.rmse.to_numpy(float); grand = y.mean(); sst = ((y-grand)**2).sum()
def dummies(col):
    d = pd.get_dummies(df[col], drop_first=True).to_numpy(float)
    return d
parts = {}
X = np.column_stack([np.ones(len(df)), dummies('cell'), dummies('valset')])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
fit_full = X @ beta
ss_full = ((fit_full-grand)**2).sum()
Xc = np.column_stack([np.ones(len(df)), dummies('valset')])
bc, *_ = np.linalg.lstsq(Xc, y, rcond=None)
ss_valset = ((Xc@bc-grand)**2).sum()
part_cell = ss_full - ss_valset
resid = sst - ss_full
print(f'runs={len(df)} cells={df.cell.nunique()} valsets={df.valset.nunique()}')
print(f'  val-set  : {100*ss_valset/sst:5.1f}%')
print(f'  cell     : {100*part_cell/sst:5.1f}%')
print(f'  residual (seed + unbalanced): {100*resid/sst:5.1f}%')
print()
print('per-valset mean rmse:', df.groupby('valset').rmse.mean().round(2).to_dict())
print('per-cell mean rmse  :', df.groupby('cell').rmse.mean().round(2).sort_values().to_dict())
