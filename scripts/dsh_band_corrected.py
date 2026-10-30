"""Correct decomposition: bias vs variance by RUL band, separating capped targets."""
import glob
import numpy as np, pandas as pd
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
d['e'] = np.clip(d.y_pred,0,125) - d.y_true
u = d[d.y_true < 124.5].copy()
print(f"{'RUL band':10s} {'n':>4s} {'bias':>8s} {'sd(e)':>7s} {'late':>6s} {'RMSE':>7s}")
for lo, hi in ((0,30),(30,60),(60,90),(90,120)):
    s = u[(u.y_true>=lo)&(u.y_true<hi)]
    if len(s)==0: continue
    print(f'{str(lo)+"-"+str(hi):10s} {len(s):4d} {s.e.mean():+8.2f} {s.e.std():7.2f} {(s.e>0).mean():6.3f} {np.sqrt((s.e**2).mean()):7.2f}')
print()
print('capped-target group (y_true == 125):')
c = d[d.y_true >= 124.5]
print(f'  n={len(c)} bias={c.e.mean():+.2f} sd={c.e.std():.2f} late={(c.e>0).mean():.3f}')
