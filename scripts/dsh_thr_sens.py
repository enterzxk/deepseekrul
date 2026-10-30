"""Threshold sensitivity of the near-failure effect (garden-of-forking-paths check)."""
import glob
import numpy as np, pandas as pd
ROOTS = {'split7':'order03','split8':'split8','split9':'split9'}
def load(root, tag):
    fs = glob.glob(f'experiments/{root}/{tag}/FD002/*/protocol_c_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float) ** 2
    return d
DATA = {}
for name, root in ROOTS.items():
    a, b = load(root,'M0_raw_tf'), load(root,'M1_global')
    if a is None or b is None: continue
    DATA[name] = (a, b)
print(' thr  engines  M0     M1     diff    CI             sig')
for thr in (15, 20, 25, 30, 40, 50, 60, 90, 126):
    rows = []
    for name, (a, b) in DATA.items():
        x = a[a.y_true <= thr].groupby('unit_id').sq.mean()
        y = b[b.y_true <= thr].groupby('unit_id').sq.mean()
        j = pd.concat([x.rename('m0'), y.rename('m1')], axis=1).dropna()
        if len(j) >= 5: rows.append(j)
    if not rows: continue
    J = pd.concat(rows)
    x, y = J.m0.to_numpy(), J.m1.to_numpy()
    rng = np.random.default_rng(20261022); d = []
    for _ in range(3000):
        t = rng.integers(0, len(x), len(x))
        d.append(np.sqrt(y[t].mean()) - np.sqrt(x[t].mean()))
    d = np.asarray(d)
    sig = bool(np.percentile(d,2.5) > 0 or np.percentile(d,97.5) < 0)
    print(f'{thr:4d} {len(J):8d} {np.sqrt(x.mean()):6.2f} {np.sqrt(y.mean()):6.2f} {np.sqrt(y.mean())-np.sqrt(x.mean()):+7.2f}  [{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}]  {sig}')
