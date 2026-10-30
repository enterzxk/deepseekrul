"""Standardise the metric: clip predictions to [0,125], recompute all cells."""
import glob
import numpy as np, pandas as pd
rows = []
for tag in ('M1_global','M6_tfA','M0_raw_tf','M4_racdr','M4b_popgate','M5_globalgate','M7_rawback'):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: continue
    d = pd.concat([pd.read_csv(f) for f in fs])
    y = d.y_true.to_numpy(float); p = np.clip(d.y_pred.to_numpy(float), 0, 125)
    e = p - y
    nasa = float(np.expm1(np.abs(e)/13).mean() - 1)
    rows.append({'cell': tag, 'n': len(y), 'rmse': float(np.sqrt((e**2).mean())),
                 'mae': float(np.abs(e).mean()), 'nasa': nasa,
                 'late_fraction': float((e>0).mean()), 'severe_late': float((e>30).mean())})
t = pd.DataFrame(rows).sort_values('rmse')
t.to_csv('results/dsh/review_r1/metrics_clipped_round146.csv', index=False)
print(f"{'cell':14s} {'n':>4s} {'RMSE':>7s} {'MAE':>6s} {'NASA':>7s} {'late':>6s} {'severe':>7s}")
for _, r in t.iterrows():
    print(f"{r.cell:14s} {r.n:4d} {r.rmse:7.3f} {r.mae:6.2f} {r.nasa:7.1f} {r.late_fraction:6.3f} {r.severe_late:7.3f}")
