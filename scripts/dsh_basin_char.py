"""Characterise the basins: bias, spread, late fraction."""
import glob
import numpy as np, pandas as pd
print(f"{'cell':14s} {'RMSE':>6s} {'bias':>7s} {'pred mean':>9s} {'pred sd':>7s} {'late':>6s} {'severe':>7s}")
for tag in ('M1_global','M6_tfA','M0_raw_tf','M4_racdr','M5_globalgate','M4b_popgate'):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: continue
    d = pd.concat([pd.read_csv(f) for f in fs])
    e = d.signed_error.astype(float)
    late = float((e > 0).mean())
    sev = float((e > 30).mean())
    print(f'{tag:14s} {np.sqrt((e**2).mean()):6.2f} {e.mean():+7.2f} {d.y_pred.mean():9.2f} {d.y_pred.std():7.2f} {late:6.3f} {sev:7.3f}')
