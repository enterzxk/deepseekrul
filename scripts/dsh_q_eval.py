import glob
import numpy as np, pandas as pd
from pathlib import Path
def load(tag):
    fs = glob.glob(f'experiments/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    return d.groupby('unit_id').agg(y=('y_true','mean'), p=('y_pred','mean'))
base = load('order03/M1_global')
q = load('q025')
truth = base.y
print('model              rmse   late  severe  early>10  bias')
for name, g in (('M1 (mean loss)', base), ('q025 (pinball 0.25)', q)):
    j = g.join(truth.rename('t'), rsuffix='_b', how='inner')
    e = j.p - j.t
    print(f'{name:20s} {np.sqrt((e**2).mean()):6.2f} {(e>0).mean():6.3f} {(e>10).sum():6d} {(e<-10).mean():9.3f} {e.mean():+6.2f}')
j = pd.concat([base.p.rename('m1'), q.p.rename('q')], axis=1).dropna()
print()
print('prediction correlation with baseline:', round(float(j.corr().iloc[0,1]), 4))
print('pre-registered gate: r < 0.95 required to be distinguishable')
