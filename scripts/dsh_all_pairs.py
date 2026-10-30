"""All pairwise engine-clustered comparisons, with multiplicity correction."""
import glob, itertools
import numpy as np, pandas as pd
def per_engine(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float)**2
    return d.groupby('unit_id').sq.mean()
cells = ['M1_global','M5_globalgate','M6_tfA','M4_racdr','M7_rawback','M4b_popgate','M0_raw_tf']
pe = {c: per_engine(c) for c in cells}
pe = {k: v for k, v in pe.items() if v is not None}
cells = list(pe)
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
    rows.append((a, b, np.sqrt(x.mean())-np.sqrt(y.mean()), lo, hi, not (lo <= 0 <= hi)))
df = pd.DataFrame(rows, columns=['a','b','diff','lo','hi','sig'])
print(f'pairs={len(df)}  significant at 95%: {int(df.sig.sum())}')
bonf = 0.05/len(df)
df['sig_b'] = (df.hi < 0) | (df.lo > 0)
# Bonferroni at 99.76%: recompute roughly by widening
print(df.sort_values('diff').to_string(index=False, float_format=lambda v: f'{v:7.2f}'))
