"""Selection regret and the power of the near-failure comparison."""
import glob
import numpy as np, pandas as pd
def near(tag, thr=30):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None, None
    d = pd.concat([pd.read_csv(f) for f in fs])
    g = d.groupby('unit_id').agg(y=('y_true','mean'), p=('y_pred','mean'))
    g['e'] = g.p - g.y
    return g[g.y <= thr].e, g
e0, g0 = near('M0_raw_tf'); e1, g1 = near('M1_global')
print(f'near engines: M0={len(e0)} M1={len(e1)}')
for nm, e in (('M0_raw_tf', e0), ('M1_global', e1)):
    print(f'{nm:12s} near_rmse={np.sqrt((e**2).mean()):.2f} near_mae={e.abs().mean():.2f}')
print()
print('selection regret: choosing by GLOBAL rmse (M1) vs best NEAR-FAILURE (M0)')
r0 = np.sqrt((e0**2).mean()); r1 = np.sqrt((e1**2).mean())
print(f'  near-failure RMSE: {r1:.2f} vs {r0:.2f}  -> +{r1-r0:.2f} ({100*(r1-r0)/r0:.0f}% worse)')
print()
# pooling near-failure engines from both splits is not valid; bootstrap within the union of engines
union = sorted(set(e0.index) | set(e1.index))
print('bootstrap of the near-failure difference (unpaired, independent resampling):')
rng = np.random.default_rng(20261022); diffs = []
for _ in range(5000):
    a = e0.to_numpy()[rng.integers(0, len(e0), len(e0))]
    b = e1.to_numpy()[rng.integers(0, len(e1), len(e1))]
    diffs.append(np.sqrt((b**2).mean()) - np.sqrt((a**2).mean()))
d = np.asarray(diffs)
print(f'  diff={np.median(d):+.2f} CI=[{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}] sig={bool(np.percentile(d,2.5)>0 or np.percentile(d,97.5)<0)}')
