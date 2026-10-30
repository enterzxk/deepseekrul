"""Capacity-matched test of H1: M7_rawback (41.6k, no features) vs M1_global (41.9k, features)."""
import glob
import numpy as np, pandas as pd
ROOTS = {'split7':'order03','split8':'split8','split9':'split9'}
def load(root, tag):
    fs = glob.glob(f'experiments/{root}/{tag}/FD002/*/protocol_c_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float) ** 2
    return d
print('capacity-matched: M7 (raw, 41.6k) vs M1 (features, 41.9k)')
print(' thr  engines   M7     M1     M7-M1      CI            sig')
for thr in (15, 20, 25, 30, 40, 60, 126):
    rows = []
    for name, root in ROOTS.items():
        a, b = load(root,'M7_rawback'), load(root,'M1_global')
        if a is None or b is None: continue
        x = a[a.y_true<=thr].groupby('unit_id').sq.mean()
        y = b[b.y_true<=thr].groupby('unit_id').sq.mean()
        j = pd.concat([x.rename('m7'), y.rename('m1')], axis=1).dropna()
        if len(j) >= 5: rows.append(j)
    if not rows: print(f'{thr:4d}  missing'); continue
    J = pd.concat(rows); x, y = J.m7.to_numpy(), J.m1.to_numpy()
    rng = np.random.default_rng(20261022); d = []
    for _ in range(3000):
        t = rng.integers(0, len(x), len(x))
        d.append(np.sqrt(x[t].mean()) - np.sqrt(y[t].mean()))
    d = np.asarray(d)
    sig = bool(np.percentile(d,2.5) > 0 or np.percentile(d,97.5) < 0)
    print(f'{thr:4d} {len(J):8d} {np.sqrt(x.mean()):6.2f} {np.sqrt(y.mean()):6.2f} {np.sqrt(x.mean())-np.sqrt(y.mean()):+8.2f} [{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}]  {sig}')
