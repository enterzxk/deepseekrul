"""Protocol C: window-level (n=windows) vs engine-clustered (n=engines)."""
import glob
import numpy as np, pandas as pd
def load(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/protocol_c_predictions.csv')
    if not fs: return None
    return pd.concat([pd.read_csv(f) for f in fs])
a, b = load('M1_global'), load('M6_tfA')
ka = a.groupby('unit_id').signed_error.apply(lambda s: (s**2).mean())
kb = b.groupby('unit_id').signed_error.apply(lambda s: (s**2).mean())
idx = ka.index.intersection(kb.index)
x, y = ka.loc[idx].to_numpy(), kb.loc[idx].to_numpy()
wa = a.signed_error.to_numpy()**2; wb = b.signed_error.to_numpy()**2
print(f'engines={len(idx)} windows={len(wa)} (x{len(wa)/len(idx):.1f} per engine)')
rng = np.random.default_rng(20261022)
d1 = []
for _ in range(5000):
    t = rng.integers(0, len(x), len(x))
    d1.append(np.sqrt(x[t].mean()) - np.sqrt(y[t].mean()))
d1 = np.asarray(d1)
d2 = []
for _ in range(5000):
    t1 = rng.integers(0, len(wa), len(wa)); t2 = rng.integers(0, len(wb), len(wb))
    d2.append(np.sqrt(wa[t1].mean()) - np.sqrt(wb[t2].mean()))
d2 = np.asarray(d2)
for name, d in (('engine-clustered', d1), ('window-level', d2)):
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f'  {name:16s} diff={np.median(d):+.2f} CI=[{lo:+.2f},{hi:+.2f}] width={hi-lo:5.2f} sig={bool(lo>0 or hi<0)}')
