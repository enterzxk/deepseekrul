"""All key comparisons under the most sensitive evaluation available:"""
"""per-engine RMSE over all Protocol-C cutoffs + engine-clustered bootstrap."""
import glob
from pathlib import Path
import numpy as np, pandas as pd

ROOTS = {'split7': 'order03', 'split8': 'split8', 'split9': 'split9'}

def per_engine(root, tag):
    fs = glob.glob(f'experiments/{root}/{tag}/FD002/*/protocol_c_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float) ** 2
    return d.groupby('unit_id').sq.mean().pow(0.5)

def boot(a, b, reps=5000, seed=20261022):
    idx = a.index.intersection(b.index)
    if len(idx) < 8: return None
    x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    rng = np.random.default_rng(seed); d = []
    for _ in range(reps):
        t = rng.integers(0, len(x), len(x))
        d.append(np.sqrt((x[t]**2).mean()) - np.sqrt((y[t]**2).mean()))
    d = np.asarray(d)
    return dict(n=len(idx), a=float(np.sqrt((x**2).mean())), b=float(np.sqrt((y**2).mean())),
                diff=float(np.median(d)), lo=float(np.percentile(d,2.5)), hi=float(np.percentile(d,97.5)),
                p=float((d<0).mean()), sig=bool(np.percentile(d,2.5)>0 or np.percentile(d,97.5)<0))

pairs = [('M1_global','M6_tfA'), ('M6_tfA','M0_raw_tf'), ('M1_global','M0_raw_tf')]
for a_tag, b_tag in pairs:
    print(f'=== {a_tag} vs {b_tag} (protocol C, engine-clustered)')
    for name, root in ROOTS.items():
        a, b = per_engine(root, a_tag), per_engine(root, b_tag)
        if a is None or b is None:
            print(f'  {name}: missing'); continue
        r = boot(a, b)
        if r:
            print(f"  {name}: n={r['n']} {r['a']:.2f} vs {r['b']:.2f} diff={r['diff']:+.2f} "
                  f"CI=[{r['lo']:+.2f},{r['hi']:+.2f}] p={r['p']:.3f} sig={r['sig']}")
