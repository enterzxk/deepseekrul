"""Is the 90-120 band over-prediction due to saturation toward the cap?"""
import glob
import numpy as np, pandas as pd
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
d['p'] = np.clip(d.y_pred, 0, 125)
u = d[d.y_true < 124.5]
s = u[(u.y_true>=90)&(u.y_true<120)]
print(f'band 90-120: n={len(s)}')
print(f'  target: mean={s.y_true.mean():.1f} range=[{s.y_true.min():.0f},{s.y_true.max():.0f}]')
print(f'  pred  : mean={s.p.mean():.1f} sd={s.p.std():.1f} range=[{s.p.min():.0f},{s.p.max():.0f}]')
print(f'  fraction of preds in [115,125]: {((s.p>=115)&(s.p<=125)).mean():.3f}')
print(f'  fraction of preds > 125 (clipped): {(s.y_pred>125).mean():.3f}')
print()
print('all windows: pred distribution')
for lo, hi in ((0,20),(20,40),(40,60),(60,80),(80,100),(100,125)):
    m = (d.p>=lo)&(d.p<hi)
    print(f'  pred {lo:3d}-{hi:3d}: {(m).sum():4d} windows ({100*m.mean():5.1f}%)')
