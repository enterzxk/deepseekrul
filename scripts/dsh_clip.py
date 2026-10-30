"""Effect of clipping predictions to [0, cap] across cells."""
import glob
import numpy as np, pandas as pd
print(f"{'cell':14s} {'raw RMSE':>9s} {'clip RMSE':>10s} {'raw late':>9s} {'clip late':>10s} {'clip severe':>12s}")
for tag in ('M1_global','M6_tfA','M0_raw_tf','M4_racdr','M5_globalgate','M4b_popgate'):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: continue
    d = pd.concat([pd.read_csv(f) for f in fs])
    y = d.y_true.to_numpy(float); p = d.y_pred.to_numpy(float)
    pc = np.clip(p, 0, 125)
    e, ec = p - y, pc - y
    print(f'{tag:14s} {np.sqrt((e**2).mean()):9.3f} {np.sqrt((ec**2).mean()):10.3f} {(e>0).mean():9.3f} {(ec>0).mean():10.3f} {(ec>30).mean():12.3f}')
