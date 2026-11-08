"""D6: engine-clustered bootstrap on the ensemble predictions, four subsets."""
import glob
import numpy as np, pandas as pd
from pathlib import Path
C = Path('/home/user/rulb/experiments/TESTCACHE')
REPS = 5000
rng = np.random.default_rng(20261108)
def nasa(e): return np.where(e > 0, np.expm1(e/10.0), np.expm1(-e/13.0))

print('%-6s %5s %8s %8s %8s %8s %8s' % ('subset','n_eng','RMSE','MAE','NASA','bias','CI_lo','CI_hi'))
store = {}
for subset in ['FD001','FD002','FD003','FD004']:
    fs = sorted(C.glob(f'{subset}_s*_f*.csv'))
    if not fs: continue
    ds = [pd.read_csv(f).sort_values('unit').reset_index(drop=True) for f in fs]
    y = ds[0].y_true.to_numpy()
    P = np.stack([d.y_pred.to_numpy() for d in ds]).mean(axis=0)
    e = P - y
    g = pd.DataFrame({'u': ds[0].unit, 'e2': e**2, 'ae': np.abs(e), 's': nasa(e), 'b': e}).groupby('u').mean()
    rmse = float(np.sqrt(g.e2.mean())); mae = float(g.ae.mean()); ns = float(g.s.mean()); bi = float(g.b.mean())
    idx = rng.integers(0, len(g), (REPS, len(g)))
    boot = np.sqrt(g.e2.to_numpy()[idx].mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    store[subset] = (rmse, mae, ns, bi, lo, hi, len(g))
    print('%-6s %5d %8.3f %8.3f %8.2f %+8.3f %8.3f %8.3f' % (subset, len(g), rmse, mae, ns, bi, lo, hi))

print()
tot = sum(v[6] for v in store.values()); w = np.array([v[6] for v in store.values()]) / tot
r = np.array([v[0] for v in store.values()])
print('pooled over subsets (engine-weighted): %.4f' % float((w * r).sum()))
print('mean RMSE over subsets:              %.4f' % float(r.mean()))