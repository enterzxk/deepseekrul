"""Is the per-unit paired-difference SD comparable between FD001 and FD002?"""
import glob
import numpy as np, pandas as pd
def per_seed(sub, tag, kind='matched_cutoff_predictions.csv'):
    fs = glob.glob(f'experiments/order03/{tag}/{sub}/*/{kind}')
    if not fs: return None
    out = []
    for f in fs:
        d = pd.read_csv(f); d['sq'] = d.signed_error.astype(float) ** 2
        out.append(d.groupby('unit_id').sq.mean().rename(f))
    return pd.concat(out, axis=1).mean(axis=1)
for sub in ('FD001','FD002'):
    a = per_seed(sub,'M1_global'); b = per_seed(sub,'M0_raw_tf')
    if a is None or b is None:
        print(sub,'missing'); continue
    j = pd.concat([a.rename('m1'), b.rename('m0')], axis=1).dropna()
    diff = j.m0 - j.m1
    rmse = np.sqrt(j.m1.mean())
    print(f'{sub}: n={len(j)} rmse(M1)={rmse:.2f} SD(diff)={diff.std():.1f} corr={j.m1.corr(j.m0):.3f}')
