"""Post-hoc shift baseline: what does a trivial constant shift achieve?"""
import glob
import numpy as np, pandas as pd
from pathlib import Path
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
g = d.groupby('unit_id').agg(y=('y_true','mean'), p=('y_pred','mean'))
y, p = g.y.to_numpy(float), g.p.to_numpy(float)
print(f'base: rmse={np.sqrt(((p-y)**2).mean()):.2f} late={(p>y).mean():.3f} severe={(p-y>10).sum()}')
print()
print(' shift   rmse   late  severe  early>10')
for s in (0, -2, -4, -6, -8, -10, -12):
    q = p + s; e = q - y
    print(f'{s:6d} {np.sqrt((e**2).mean()):6.2f} {(e>0).mean():6.3f} {(e>10).sum():6d} {(e<-10).mean():7.3f}')
