"""How much does pooling inflate the difference SD? Calibrates the MDE table."""
import glob
import numpy as np, pandas as pd
def per_engine_single(cell, root='order03'):
    frames = []
    for f in glob.glob(f'experiments/{root}/{cell}/FD002/*/matched_cutoff_predictions.csv'):
        d = pd.read_csv(f); d['sq'] = d.signed_error.astype(float)**2
        frames.append(d.groupby('unit_id').sq.mean())
    return pd.concat(frames, axis=1).mean(axis=1)
def per_engine_pooled(cell):
    frames = []
    for root in ('order03','split8','split9'):
        for f in glob.glob(f'experiments/{root}/{cell}/FD002/*/matched_cutoff_predictions.csv'):
            d = pd.read_csv(f); d['sq'] = d.signed_error.astype(float)**2
            frames.append(d.groupby('unit_id').sq.mean())
    return pd.concat(frames, axis=1).mean(axis=1)
for pair in (('M1_global','M6_tfA'), ('M1_global','M0_raw_tf')):
    a, b = pair
    s1a, s1b = per_engine_single(a), per_engine_single(b)
    j1 = pd.concat([s1a.rename('a'), s1b.rename('b')], axis=1).dropna()
    d1 = (j1.b - j1.a).to_numpy()
    pa, pb = per_engine_pooled(a), per_engine_pooled(b)
    j2 = pd.concat([pa.rename('a'), pb.rename('b')], axis=1).dropna()
    d2 = (j2.b - j2.a).to_numpy()
    print(f'{a} vs {b}:')
    print(f'  single split n={len(d1):3d}  SD(diff)={d1.std(ddof=1):7.1f}')
    print(f'  pooled      n={len(d2):3d}  SD(diff)={d2.std(ddof=1):7.1f}   ratio={d2.std(ddof=1)/d1.std(ddof=1):.2f}')
