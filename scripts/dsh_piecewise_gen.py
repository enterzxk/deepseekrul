"""Does the piecewise correction generalise? FD001, M6, Protocol C."""
import glob
import numpy as np, pandas as pd
GRID = [(t,dl,dh) for t in (40,60,80) for dl in (-10,-6,-3) for dh in (0,-2)]
def apply(p, thr, dl, dh):
    q = p.copy(); m = q < thr; q[m] += dl; q[~m] += dh
    return np.clip(q, 0, 125)
def score(p, y):
    e = np.clip(p,0,125) - y
    return float(np.sqrt((e**2).mean())), float((e>0).mean())
def load(cell, subset, kind):
    fs = glob.glob(f'experiments/order03/{cell}/{subset}/*/{kind}')
    if not fs: return None
    return pd.concat([pd.read_csv(f) for f in fs])
cases = [('M1_global','FD002','matched_cutoff_predictions.csv'),
         ('M1_global','FD001','matched_cutoff_predictions.csv'),
         ('M6_tfA','FD002','matched_cutoff_predictions.csv'),
         ('M0_raw_tf','FD002','matched_cutoff_predictions.csv'),
         ('M1_global','FD002','protocol_c_predictions.csv')]
print(f"{'cell/subset/protocol':38s} {'n':>5s} {'base RMSE':>9s} {'base late':>9s} {'adj RMSE':>8s} {'adj late':>8s} {'dLate':>7s}")
for cell, sub, kind in cases:
    d = load(cell, sub, kind)
    if d is None or len(d) < 20: print(f'{cell}/{sub}/{kind[:10]} missing'); continue
    y = d.y_true.to_numpy(float); p = np.clip(d.y_pred.to_numpy(float), 0, 125)
    r0, l0 = score(p, y)
    best = None
    for thr, dl, dh in GRID:
        r, l = score(apply(p, thr, dl, dh), y)
        if l < l0 and (best is None or r < best[0]): best = (r, l, thr, dl, dh)
    if best is None:
        print(f'{cell}/{sub}/{kind[:10]:10s} no correction reduced late'); continue
    tag = f'{cell}/{sub}/{kind.split("_")[0]}'
    print(f'{tag:38s} {len(y):5d} {r0:9.3f} {l0:9.3f} {best[0]:8.3f} {best[1]:8.3f} {best[1]-l0:+7.3f}')
