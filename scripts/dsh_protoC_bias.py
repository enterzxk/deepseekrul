"""Protocol C: bias by band and by cutoff position."""
import glob
import numpy as np, pandas as pd
fs = glob.glob('experiments/order03/M1_global/FD002/*/protocol_c_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
d['p'] = np.clip(d.y_pred, 0, 125)
d['e'] = d.p - d.y_true
print('fraction of targets at the cap, by cutoff:')
for c in sorted(d.abs_cut.unique()):
    s = d[np.isclose(d.abs_cut, c)]
    cap = (s.y_true >= 124.5).mean()
    print(f'  cutoff {c:5.0f}: n={len(s):3d} targets_at_125={cap:.2f} bias={s.e.mean():+7.2f}')
print()
print('bias by RUL band (uncapped targets only):')
u = d[d.y_true < 124.5]
for lo, hi in ((0,30),(30,60),(60,90),(90,120)):
    s = u[(u.y_true>=lo)&(u.y_true<hi)]
    if len(s): print(f'  RUL {lo:3d}-{hi:3d}: n={len(s):4d} bias={s.e.mean():+7.2f} late={(s.e>0).mean():.3f}')
