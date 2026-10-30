"""Two-segment asymmetric correction: reduce late risk at low predicted RUL."""
import glob
import numpy as np, pandas as pd
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
y = d.y_true.to_numpy(float); p0 = np.clip(d.y_pred.to_numpy(float), 0, 125)
def score(p):
    q = np.clip(p, 0, 125); e = q - y
    return np.sqrt((e**2).mean()), (e>0).mean(), (e>30).mean()
r0, l0, s0 = score(p0)
print(f'baseline            RMSE={r0:6.3f} late={l0:.3f} severe={s0:.3f}')
print()
print('threshold  delta_low  delta_high   RMSE    late  severe')
best = None
for thr in (40, 60, 80):
    for dl in (-10, -6, -3):
        for dh in (0, -2):
            p = p0.copy()
            m = p < thr
            p[m] += dl
            p[~m] += dh
            r, l, s = score(p)
            if l < l0 and (best is None or r < best[0]):
                best = (r, l, s, thr, dl, dh)
            if l < l0 - 0.03:
                print(f'{thr:9d} {dl:10d} {dh:11d} {r:7.3f} {l:6.3f} {s:7.3f}')
print()
if best: print(f'best RMSE among late-reducing: RMSE={best[0]:.3f} late={best[1]:.3f} (thr={best[3]}, dl={best[4]}, dh={best[5]})')
