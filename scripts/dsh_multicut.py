"""Does using all Protocol-C cutoffs per engine tighten the comparison?"""
import glob
from pathlib import Path
import numpy as np, pandas as pd

def per_engine(tag, kind='protocol_c_predictions.csv'):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/{kind}')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float) ** 2
    return d.groupby('unit_id').sq.mean().pow(0.5)

def boot(a, b, reps=5000, seed=20261022):
    idx = a.index.intersection(b.index)
    x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    rng = np.random.default_rng(seed); d = []
    for _ in range(reps):
        t = rng.integers(0, len(x), len(x))
        d.append(np.sqrt((x[t]**2).mean()) - np.sqrt((y[t]**2).mean()))
    d = np.asarray(d)
    w = np.percentile(d, 97.5) - np.percentile(d, 2.5)
    return len(idx), float(np.median(d)), float(np.percentile(d,2.5)), float(np.percentile(d,97.5)), float(w)

for kind, label in ((('matched_cutoff_predictions.csv'), 'matched (1 point/engine)'),
                    (('protocol_c_predictions.csv'), 'protocol C (5 points/engine)')):
    a, b = per_engine('M1_global', kind), per_engine('M6_tfA', kind)
    if a is None or b is None:
        print(label, 'missing'); continue
    n, med, lo, hi, w = boot(a, b)
    print(f'{label:28s} n={n} diff={med:+.3f} CI=[{lo:+.3f},{hi:+.3f}] width={w:.3f}')
