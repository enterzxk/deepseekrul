"""Does a single constant offset beat the models? (bias-correction baseline)"""
import glob
import numpy as np, pandas as pd
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
y, p = d.y_true.to_numpy(float), d.y_pred.to_numpy(float)
print(f"{'delta':>6s} {'RMSE':>7s} {'MAE':>6s} {'late':>6s} {'severe':>7s}")
for dlt in (0, -1, -2, -3, -4, -5, -6, -8, 2):
    q = np.clip(p + dlt, 0, 125)
    e = q - y
    print(f'{dlt:+6.0f} {np.sqrt((e**2).mean()):7.3f} {np.abs(e).mean():6.2f} {(e>0).mean():6.3f} {(e>30).mean():7.3f}')
