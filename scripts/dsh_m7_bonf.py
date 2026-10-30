"""Does the single nominal hit survive multiplicity correction?"""
import glob
import numpy as np, pandas as pd
def per_engine(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float)**2
    return d.groupby('unit_id').sq.mean()
a, b = per_engine('M1_global'), per_engine('M7_rawback')
j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
x, y = j.a.to_numpy(), j.b.to_numpy()
rng = np.random.default_rng(20261022)
ds = []
for _ in range(20000):
    t = rng.integers(0, len(x), len(x))
    ds.append(np.sqrt(x[t].mean()) - np.sqrt(y[t].mean()))
ds = np.asarray(ds)
m = 21; alpha = 0.05
lo95, hi95 = np.percentile(ds, [2.5, 97.5])
lo_b, hi_b = np.percentile(ds, [100*alpha/(2*m), 100*(1-alpha/(2*m))])
print(f'n={len(x)}  diff={np.median(ds):+.2f}')
print(f'  uncorrected 95%   [{lo95:+.2f}, {hi95:+.2f}]  sig={not (lo95<=0<=hi95)}')
print(f'  Bonferroni (m=21)[{lo_b:+.2f}, {hi_b:+.2f}]  sig={not (lo_b<=0<=hi_b)}')
print()
print('interpretation: 1/21 nominal hits is exactly the count expected by chance (21*0.05=1.05)')
