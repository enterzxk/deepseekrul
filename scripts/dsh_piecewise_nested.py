"""Nested validation of the piecewise correction."""
import glob
import numpy as np, pandas as pd
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
units = d.unit_id.to_numpy()
y = d.y_true.to_numpy(float); p0 = np.clip(d.y_pred.to_numpy(float), 0, 125)
grid = [(thr, dl, dh) for thr in (40,60,80) for dl in (-10,-6,-3) for dh in (0,-2)]
def score(p, yy):
    q = np.clip(p, 0, 125); e = q - yy
    return float(np.sqrt((e**2).mean())), float((e>0).mean())
rng = np.random.default_rng(20261022)
u = np.unique(units)
gains_r, gains_l = [], []
for _ in range(200):
    perm = rng.permutation(u); h = len(perm)//2
    A, B = set(perm[:h]), set(perm[h:])
    mA = np.isin(units, list(A)); mB = ~mA
    # choose on A: minimise RMSE subject to late < baseline late
    rA0, lA0 = score(p0[mA], y[mA])
    best = None
    for thr, dl, dh in grid:
        p = p0[mA].copy(); m = p < thr; p[m] += dl; p[~m] += dh
        r, l = score(p, y[mA])
        if l < lA0 and (best is None or r < best[0] or (abs(r-best[0])<1e-9 and l < best[1])):
            best = (r, l, thr, dl, dh)
    thr, dl, dh = best[2], best[3], best[4]
    p = p0[mB].copy(); m = p < thr; p[m] += dl; p[~m] += dh
    rB, lB = score(p, y[mB]); rB0, lB0 = score(p0[mB], y[mB])
    gains_r.append(rB - rB0); gains_l.append(lB - lB0)
gr, gl = np.asarray(gains_r), np.asarray(gains_l)
print(f'200 nested splits')
print(f'  RMSE change : median {np.median(gr):+.3f}  95% [{np.percentile(gr,2.5):+.3f}, {np.percentile(gr,97.5):+.3f}]')
print(f'  late change : median {np.median(gl):+.3f}  95% [{np.percentile(gl,2.5):+.3f}, {np.percentile(gl,97.5):+.3f}]')
print(f'  P(late improves) = {float((gl<0).mean()):.2f}   P(RMSE within +0.3) = {float((gr<0.3).mean()):.2f}')
