import json, glob, hashlib
import numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob('experiments/*/*/FD002/*/metrics.json')):
    m = json.load(open(f)); vu = m.get('val_units') or []
    if not vu: continue
    h = hashlib.md5(','.join(map(str, sorted(vu))).encode()).hexdigest()[:8]
    p = f.split('/')
    rows.append({'cell': p[2], 'valset': h, 'rmse': m.get('matched_cutoff_rmse')})
df = pd.DataFrame(rows).dropna()
keep = ['M0_raw_tf','M1_global','M6_tfA']
sub = df[df.cell.isin(keep)]
piv = sub.groupby(['cell','valset']).rmse.mean().unstack()
print(piv.round(2).to_string())
print()
y = piv.to_numpy().ravel(); y = y[~np.isnan(y)]
grand = y.mean(); sst = ((y-grand)**2).sum()
cm = piv.mean(axis=1); vm = piv.mean(axis=0); em = piv - cm.values[:,None] - vm.values[None,:] + grand
ss_cell = float((piv.notna().sum(axis=1) * (cm - grand)**2).sum())
ss_val  = float((piv.notna().sum(axis=0) * (vm - grand)**2).sum())
resid = piv - cm.values[:,None] - vm.values[None,:] + grand
ss_res = float(np.nansum(resid.to_numpy()**2))
print(f'balanced 3x3: cell {100*ss_cell/sst:.1f}%  valset {100*ss_val/sst:.1f}%  resid {100*ss_res/sst:.1f}%')
print('cell means:', cm.round(2).to_dict())
print('valset means:', vm.round(2).to_dict())
