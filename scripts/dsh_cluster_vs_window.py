"""Same comparison, two resampling schemes: engine-clustered vs window-level."""
import glob
import numpy as np, pandas as pd
def load(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    return pd.concat([pd.read_csv(f) for f in fs])
a, b = load('M1_global'), load('M6_tfA')
ka = a.groupby('unit_id').signed_error.apply(lambda s: (s**2).mean())
kb = b.groupby('unit_id').signed_error.apply(lambda s: (s**2).mean())
idx = ka.index.intersection(kb.index)
x, y = ka.loc[idx].to_numpy(), kb.loc[idx].to_numpy()
print(f'engines={len(idx)}')
print(f'  engine RMSE: M1={np.sqrt(x.mean()):.2f} M6={np.sqrt(y.mean()):.2f} diff={np.sqrt(x.mean())-np.sqrt(y.mean()):+.2f}')
rng = np.random.default_rng(20261022)
# engine-clustered
d1 = []
for _ in range(5000):
    t = rng.integers(0, len(x), len(x))
    d1.append(np.sqrt(x[t].mean()) - np.sqrt(y[t].mean()))
d1 = np.asarray(d1)
print(f'  engine-clustered : diff={np.median(d1):+.2f} CI=[{np.percentile(d1,2.5):+.2f},{np.percentile(d1,97.5):+.2f}] sig={bool(np.percentile(d1,2.5)>0 or np.percentile(d1,97.5)<0)}')
# window-level (WRONG: treats each window as independent)
wa = a.signed_error.to_numpy()**2; wb = b.signed_error.to_numpy()**2
d2 = []
for _ in range(5000):
    t1 = rng.integers(0, len(wa), len(wa)); t2 = rng.integers(0, len(wb), len(wb))
    d2.append(np.sqrt(wa[t1].mean()) - np.sqrt(wb[t2].mean()))
d2 = np.asarray(d2)
print(f'  window-level     : diff={np.median(d2):+.2f} CI=[{np.percentile(d2,2.5):+.2f},{np.percentile(d2,97.5):+.2f}] sig={bool(np.percentile(d2,2.5)>0 or np.percentile(d2,97.5)<0)}')
print(f'  CI width: engine {np.percentile(d1,97.5)-np.percentile(d1,2.5):.2f} vs window {np.percentile(d2,97.5)-np.percentile(d2,2.5):.2f}')
