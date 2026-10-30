"""Are H2_* really identical to M4_*? Verify pairwise with a consistent join."""
import glob
import numpy as np, pandas as pd
def preds(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    return d.groupby('unit_id').y_pred.mean()
tags = ['M4_racdr','M4c_noid','H2_fixed1','H2_half','M4_ckpt','M5_globalgate','M4b_popgate']
P = {t: preds(t) for t in tags}
base = P['M4_racdr']
print(f"{'cell':14s} {'n':>3s} {'corr_with_M4':>13s} {'max|diff|':>10s}")
for t in tags:
    if P[t] is None: print(t,'missing'); continue
    j = pd.concat([base.rename('a'), P[t].rename('b')], axis=1).dropna()
    print(f'{t:14s} {len(j):3d} {j.a.corr(j.b):13.4f} {np.abs(j.a-j.b).max():10.4f}')
