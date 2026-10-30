"""Is the feature effect consistent per split? M1 (features) vs M7 (raw), same architecture."""
import glob
import numpy as np, pandas as pd
ROOTS = {'split7':'order03','split8':'split8','split9':'split9'}
def per_engine(root, tag):
    fs = glob.glob(f'experiments/{root}/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float) ** 2
    return d.groupby('unit_id').sq.mean()
print('matched-cutoff, engine-clustered, per split:')
for name, root in ROOTS.items():
    a, b = per_engine(root,'M7_rawback'), per_engine(root,'M1_global')
    if a is None or b is None:
        print(f'  {name}: missing'); continue
    j = pd.concat([a.rename('m7'), b.rename('m1')], axis=1).dropna()
    x, y = j.m7.to_numpy(), j.m1.to_numpy()
    rng = np.random.default_rng(20261022); d = []
    for _ in range(4000):
        t = rng.integers(0, len(x), len(x))
        d.append(np.sqrt(x[t].mean()) - np.sqrt(y[t].mean()))
    d = np.asarray(d)
    sig = bool(np.percentile(d,2.5) > 0 or np.percentile(d,97.5) < 0)
    print(f'  {name}: n={len(j)} M7={np.sqrt(x.mean()):.2f} M1={np.sqrt(y.mean()):.2f} M7-M1={np.sqrt(x.mean())-np.sqrt(y.mean()):+.2f} CI=[{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}] sig={sig}')
