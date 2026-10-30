"""Multiplicity on the pooled set (all splits, unique engines averaged)."""
import glob, itertools
import numpy as np, pandas as pd
ROOTS = ['order03','split8','split9']
def per_engine(cell):
    frames = []
    for root in ROOTS:
        for f in glob.glob(f'experiments/{root}/{cell}/FD002/*/matched_cutoff_predictions.csv'):
            d = pd.read_csv(f)
            d['sq'] = d.signed_error.astype(float)**2
            frames.append(d.groupby('unit_id').sq.mean())
    if not frames: return None
    # average each engine's MSE across the splits where it appears
    return pd.concat(frames, axis=1).mean(axis=1)
cells = ['M1_global','M5_globalgate','M6_tfA','M4_racdr','M7_rawback','M4b_popgate','M0_raw_tf']
pe = {c: per_engine(c) for c in cells}
pe = {k: v for k, v in pe.items() if v is not None and len(v) > 0}
cells = [c for c in cells if c in pe]
print('cells with data:', cells)
print('engine counts:', {c: len(pe[c]) for c in cells})
rng = np.random.default_rng(20261022)
rows = []
for a, b in itertools.combinations(cells, 2):
    j = pd.concat([pe[a].rename('a'), pe[b].rename('b')], axis=1).dropna()
    x, y = j.a.to_numpy(), j.b.to_numpy()
    ds = []
    for _ in range(3000):
        t = rng.integers(0, len(x), len(x))
        ds.append(np.sqrt(x[t].mean()) - np.sqrt(y[t].mean()))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    rows.append((a, b, len(x), np.sqrt(x.mean())-np.sqrt(y.mean()), lo, hi, not (lo <= 0 <= hi)))
df = pd.DataFrame(rows, columns=['a','b','n','diff','lo','hi','sig'])
print()
print(f'total pairs={len(df)}  nominal sig={int(df.sig.sum())}  (chance expectation {0.05*len(df):.1f})')
print(df.sort_values('diff').head(6).to_string(index=False, float_format=lambda v: f'{v:8.2f}'))
