"""Is the large difference-SD from low correlation, or just from M0 having larger error variance?"""
import glob
import numpy as np, pandas as pd
def per_seed(sub, tag):
    fs = glob.glob(f'experiments/order03/{tag}/{sub}/*/matched_cutoff_predictions.csv')
    if not fs: return None
    out = []
    for f in fs:
        d = pd.read_csv(f); d['sq'] = d.signed_error.astype(float) ** 2
        out.append(d.groupby('unit_id').sq.mean().rename(f))
    return pd.concat(out, axis=1).mean(axis=1)
for sub in ('FD002',):
    series = {t: per_seed(sub, t) for t in ('M1_global','M6_tfA','M0_raw_tf','M7_rawback')}
    base = series['M1_global']
    print(f'{sub}: per-engine MSE SD  (mean MSE)')
    for t, s in series.items():
        if s is None: continue
        j = pd.concat([base.rename('b'), s.rename('x')], axis=1).dropna()
        print(f'  {t:12s} SD={j.x.std():7.1f}  mean={j.x.mean():7.1f}  corr_with_M1={j.b.corr(j.x):+.3f}  SD(diff)={(j.x-j.b).std():7.1f}')
