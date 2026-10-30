"""Near-failure metrics using ALL Protocol-C windows with true RUL <= 30."""
import glob
import numpy as np, pandas as pd
def load(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/protocol_c_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float) ** 2
    return d
for tag in ('M0_raw_tf','M1_global','M6_tfA'):
    d = load(tag)
    if d is None: print(tag,'missing'); continue
    near = d[d.y_true <= 30]
    per = near.groupby('unit_id').sq.mean().pow(0.5)
    print(f'{tag:12s} windows={len(near):3d} engines={len(per):2d} near_rmse={np.sqrt((near.signed_error**2).mean()):.2f} mae={near.signed_error.abs().mean():.2f}')
print()
a = load('M0_raw_tf'); b = load('M1_global')
na = a[a.y_true<=30].groupby('unit_id').sq.mean(); nb = b[b.y_true<=30].groupby('unit_id').sq.mean()
idx = na.index.intersection(nb.index)
print('paired near-failure engines:', len(idx))
x, y = na.loc[idx].to_numpy(), nb.loc[idx].to_numpy()
rng = np.random.default_rng(20261022); d = []
for _ in range(5000):
    t = rng.integers(0, len(x), len(x))
    d.append(np.sqrt(y[t].mean()) - np.sqrt(x[t].mean()))   # x, y are MSEs -> RMSE
d = np.asarray(d)
print(f'  M1={np.sqrt(y.mean()):.2f} M0={np.sqrt(x.mean()):.2f} diff={np.median(d):+.2f} CI=[{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}] sig={bool(np.percentile(d,2.5)>0 or np.percentile(d,97.5)<0)}')
