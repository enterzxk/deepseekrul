"""Full trade-off of the correction: RMSE, MAE, NASA, late, severe, near-failure."""
import glob
import numpy as np, pandas as pd
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
y = d.y_true.to_numpy(float); p0 = np.clip(d.y_pred.to_numpy(float), 0, 125)
def apply(p, thr=40, dl=-3, dh=-2):
    q = p.copy(); m = q < thr; q[m] += dl; q[~m] += dh
    return np.clip(q, 0, 125)
def metrics(p):
    e = p - y
    nasa = float(np.expm1(np.abs(e)/13).mean() - 1)
    near = y <= 60
    return {'RMSE': np.sqrt((e**2).mean()), 'MAE': np.abs(e).mean(), 'NASA': nasa,
            'late': (e>0).mean(), 'severe': (e>30).mean(),
            'late_near': (e[near]>0).mean(), 'rmse_near': np.sqrt((e[near]**2).mean())}
a, b = metrics(p0), metrics(apply(p0))
print(f"{'metric':10s} {'baseline':>9s} {'corrected':>10s} {'change':>8s}")
for k in a:
    print(f'{k:10s} {a[k]:9.3f} {b[k]:10.3f} {b[k]-a[k]:+8.3f}')
