import glob, itertools
import numpy as np, pandas as pd
ROOTS = ['order03','split8','split9']
def per_engine(cell):
    frames = []
    for root in ROOTS:
        for f in glob.glob(f'experiments/{root}/{cell}/FD002/*/matched_cutoff_predictions.csv'):
            d = pd.read_csv(f); d['sq'] = d.signed_error.astype(float)**2
            frames.append(d.groupby('unit_id').sq.mean())
    return pd.concat(frames, axis=1).mean(axis=1) if frames else None
cells = ['M1_global','M5_globalgate','M6_tfA','M4_racdr','M7_rawback','M4b_popgate','M0_raw_tf']
pe = {c: per_engine(c) for c in cells}
pe = {k: v for k, v in pe.items() if v is not None and len(v) > 0}
cells = [c for c in cells if c in pe]
rng = np.random.default_rng(20261022)
m = len(list(itertools.combinations(cells, 2)))
out = []
for a, b in itertools.combinations(cells, 2):
    j = pd.concat([pe[a].rename('a'), pe[b].rename('b')], axis=1).dropna()
    x, y = j.a.to_numpy(), j.b.to_numpy()
    ds = np.array([np.sqrt(x[t].mean()) - np.sqrt(y[t].mean())
                   for t in (rng.integers(0, len(x), len(x)) for _ in range(4000))])
    lo95, hi95 = np.percentile(ds, [2.5, 97.5])
    loB, hiB = np.percentile(ds, [100*0.05/(2*m), 100*(1-0.05/(2*m))])
    out.append((a, b, len(x), np.sqrt(x.mean())-np.sqrt(y.mean()), lo95, hi95, loB, hiB))
df = pd.DataFrame(out, columns=['a','b','n','diff','lo95','hi95','loB','hiB'])
df['sig95'] = (df.hi95 < 0) | (df.lo95 > 0)
df['sigB']  = (df.hiB < 0) | (df.loB > 0)
print(f'pairs={len(df)}  sig95={int(df.sig95.sum())}  sigBonferroni={int(df.sigB.sum())}  chance={0.05*len(df):.1f}')
print()
print('the nominally significant pairs:')
print(df[df.sig95].to_string(index=False, float_format=lambda v: f'{v:8.2f}'))
