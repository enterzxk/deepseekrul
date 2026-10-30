"""Nested selection: pick the combo on half the engines, evaluate on the other half."""
import glob, itertools
import numpy as np, pandas as pd
def load(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    d = pd.concat([pd.read_csv(f) for f in fs])
    return d.groupby('unit_id').agg(y=('y_true','mean'), p=('y_pred','mean'))
cells = ['M1_global','M6_tfA','M5_globalgate','M4_racdr','M0_raw_tf','M7_rawback']
P = {c: load(c) for c in cells}
menu = []
for k in (1, 2, 3):
    for combo in itertools.combinations(cells, k):
        menu.append(combo)
def rmse(p, y):
    q = np.clip(p, 0, 125); return float(np.sqrt(((q-y)**2).mean()))
units = P['M1_global'].index.to_numpy()
rng = np.random.default_rng(20261022)
gains = []
chosen = {}
for _ in range(200):
    perm = rng.permutation(units)
    half = len(perm)//2
    A, B = perm[:half], perm[half:]
    best, best_r = None, np.inf
    for combo in menu:
        if any(c not in P for c in combo): continue
        pa = np.mean([P[c].p.reindex(A).to_numpy() for c in combo], axis=0)
        ya = P[combo[0]].y.reindex(A).to_numpy()
        r = rmse(pa, ya)
        if r < best_r: best, best_r = combo, r
    pb = np.mean([P[c].p.reindex(B).to_numpy() for c in best], axis=0)
    yb = P[best[0]].y.reindex(B).to_numpy()
    g = rmse(pb, yb) - rmse(P['M1_global'].p.reindex(B).to_numpy(), yb)
    gains.append(g)
    chosen[best] = chosen.get(best, 0) + 1
g = np.asarray(gains)
print(f'200 random 26/26 engine splits')
print(f'  selected-then-validated gain: mean={g.mean():+.3f}  median={np.median(g):+.3f}')
print(f'  95% CI [{np.percentile(g,2.5):+.3f}, {np.percentile(g,97.5):+.3f}]  P(gain<0)={float((g<0).mean()):.2f}')
print()
print('most frequently selected combos:')
for combo, n in sorted(chosen.items(), key=lambda kv: -kv[1])[:5]:
    print(f'  {n:4d}x  {"+".join(combo)}')
