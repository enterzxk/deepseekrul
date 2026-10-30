"""Free experiment: does cross-basin ensembling beat the best single cell?"""
import glob, itertools
import numpy as np, pandas as pd
def load(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['seed'] = d.unit_id.astype(str)
    return d.groupby('unit_id').agg(y=('y_true','mean'), p=('y_pred','mean'))
cells = ['M1_global','M5_globalgate','M6_tfA','M4_racdr','M0_raw_tf','M7_rawback']
P = {t: load(t) for t in cells}
P = {k: v for k, v in P.items() if v is not None}
def score(p, y):
    q = np.clip(p, 0, 125); e = q - y
    return np.sqrt((e**2).mean()), (e>0).mean(), (e>30).mean()
print(f"{'combination':28s} {'RMSE':>7s} {'late':>6s} {'severe':>7s}")
base = P['M1_global']
r0, l0, s0 = score(base.p.to_numpy(), base.y.to_numpy())
print(f"{'M1 alone':28s} {r0:7.3f} {l0:6.3f} {s0:7.3f}")
for k in (2, 3, 4):
    for combo in itertools.combinations(cells, k):
        if 'M1_global' not in combo: continue
        if any(c not in P for c in combo): continue
        j = pd.concat([P[c].rename(columns={'y':f'y_{c}','p':f'p_{c}'}) for c in combo], axis=1).dropna()
        p = j[[f'p_{c}' for c in combo]].mean(axis=1).to_numpy()
        y = j[f'y_{combo[0]}'].to_numpy()
        r, l, s = score(p, y)
        if r < r0:
            print(f"{'+'.join(c.replace('_global','').replace('_globalgate','G') for c in combo):28s} {r:7.3f} {l:6.3f} {s:7.3f}  <-- better")
