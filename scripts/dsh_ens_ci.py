"""Engine-clustered paired CI for M1 vs the M1+M6 ensemble."""
import glob
import numpy as np, pandas as pd
def load(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    d = pd.concat([pd.read_csv(f) for f in fs])
    return d.groupby('unit_id').agg(y=('y_true','mean'), p=('y_pred','mean'))
a, b = load('M1_global'), load('M6_tfA')
j = a.join(b, lsuffix='_m1', rsuffix='_m6').dropna()
y = j.y_m1.to_numpy()
p1 = np.clip(j.p_m1.to_numpy(), 0, 125)
p2 = np.clip(((j.p_m1 + j.p_m6)/2).to_numpy(), 0, 125)
e1, e2 = p1 - y, p2 - y
print(f'n_engines={len(y)}')
print(f'  M1 alone    RMSE={np.sqrt((e1**2).mean()):.3f}  late={(e1>0).mean():.3f}')
print(f'  M1+M6 avg   RMSE={np.sqrt((e2**2).mean()):.3f}  late={(e2>0).mean():.3f}')
rng = np.random.default_rng(20261022)
ds = []
for _ in range(10000):
    t = rng.integers(0, len(y), len(y))
    ds.append(np.sqrt((e2[t]**2).mean()) - np.sqrt((e1[t]**2).mean()))
ds = np.asarray(ds)
lo, hi = np.percentile(ds, [2.5, 97.5])
print(f'  paired diff = {np.median(ds):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  sig={bool(hi<0)}')
print(f'  relative = {100*np.median(ds)/np.sqrt((e1**2).mean()):+.1f}%')
