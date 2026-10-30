"""Near-failure comparison pooled over all three splits (engine-level, disclosed)."""
import glob
import numpy as np, pandas as pd
ROOTS = {'split7':'order03','split8':'split8','split9':'split9'}
def near_mse(root, tag, thr=30):
    fs = glob.glob(f'experiments/{root}/{tag}/FD002/*/protocol_c_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d = d[d.y_true <= thr]
    d['sq'] = d.signed_error.astype(float) ** 2
    return d.groupby('unit_id').sq.mean()
frames = {}
for name, root in ROOTS.items():
    a = near_mse(root, 'M0_raw_tf'); b = near_mse(root, 'M1_global')
    if a is None or b is None:
        print(name, 'missing'); continue
    j = pd.concat([a.rename('m0'), b.rename('m1')], axis=1).dropna()
    j['split'] = name
    frames[name] = j
    print(f'{name}: engines={len(j)} M0={np.sqrt(j.m0.mean()):.2f} M1={np.sqrt(j.m1.mean()):.2f}')
allj = pd.concat(frames.values())
print(f'pooled: {len(allj)} engine-split observations, {allj.index.nunique()} unique engines')
rng = np.random.default_rng(20261022); d = []
x = allj.m0.to_numpy(); y = allj.m1.to_numpy()
for _ in range(5000):
    t = rng.integers(0, len(x), len(x))
    d.append(np.sqrt(y[t].mean()) - np.sqrt(x[t].mean()))
d = np.asarray(d)
print(f'M1-M0 near-failure: {np.sqrt(y.mean())-np.sqrt(x.mean()):+.2f} CI=[{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}] sig={bool(np.percentile(d,2.5)>0 or np.percentile(d,97.5)<0)}')
