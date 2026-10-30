"""Is the positive bias explained by cap-125 truncation?"""
import glob
import numpy as np, pandas as pd
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
d['e'] = np.clip(d.y_pred,0,125) - d.y_true
print(f'overall bias = {d.e.mean():+.3f}  (n={len(d)})')
print()
print('by whether the target hit the 125 cap:')
for cap, s in d.groupby(d.y_true >= 124.5):
    print(f'  target {"=125 (capped)" if cap else "<125"}: n={len(s):4d} bias={s.e.mean():+7.3f}  late={(s.e>0).mean():.3f}')
print()
print('by target band (uncapped targets only):')
u = d[d.y_true < 124.5]
for lo, hi in ((0,40),(40,80),(80,120)):
    s = u[(u.y_true>=lo)&(u.y_true<hi)]
    if len(s): print(f'  RUL {lo:3d}-{hi:3d}: n={len(s):4d} bias={s.e.mean():+7.3f}')
