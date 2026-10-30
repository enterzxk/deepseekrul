"""Near-failure segment metrics per cell (true RUL <= 30), the operationally relevant part."""
import glob
import numpy as np, pandas as pd
from pathlib import Path
rows = []
for f in sorted(glob.glob('experiments/order03/*/FD002/*/matched_cutoff_predictions.csv')):
    cell = f.split('/')[2]
    d = pd.read_csv(f)
    if d.empty: continue
    g = d.groupby('unit_id').agg(y=('y_true','mean'), p=('y_pred','mean'))
    g['e'] = g.p - g.y
    near = g[g.y <= 30]
    allm = g
    if len(near) < 3: continue
    rows.append({'cell': cell, 'n_near': len(near),
                 'near_rmse': float(np.sqrt((near.e**2).mean())), 'near_mae': float(near.e.abs().mean()),
                 'near_late': float((near.e > 0).mean()), 'near_severe': int((near.e > 10).sum()),
                 'global_rmse': float(np.sqrt((allm.e**2).mean()))})
df = pd.DataFrame(rows)
agg = df.groupby('cell').mean(numeric_only=True).sort_values('near_rmse')
print(f"{'cell':16s} {'n':>3s} {'near_rmse':>9s} {'near_mae':>8s} {'near_late':>9s} {'near_sev':>8s} {'global':>7s}")
for c, r in agg.iterrows():
    print(f"{c:16s} {r.n_near:3.0f} {r.near_rmse:9.2f} {r.near_mae:8.2f} {r.near_late:9.2f} {r.near_severe:8.1f} {r.global_rmse:7.2f}")
print()
print('correlation near_rmse vs global_rmse across cells:', round(float(agg.near_rmse.corr(agg.global_rmse)), 3))
agg.to_csv('results/dsh/review_r1/near_failure_metrics.csv')
