"""Is the improving parameter set a knife-edge or a broad plateau?"""
import glob
import numpy as np, pandas as pd
THRS = (30,40,50,60,70,80); DLS = (-12,-9,-6,-3); DHS = (0,-2,-4)
def score(p, y):
    e = np.clip(p,0,125) - y
    return float(np.sqrt((e**2).mean())), float((e>0).mean())
cells = ['M1_global','M6_tfA','M0_raw_tf']
for cell in cells:
    fs = glob.glob(f'experiments/order03/{cell}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: continue
    d = pd.concat([pd.read_csv(f) for f in fs])
    y = d.y_true.to_numpy(float); p0 = np.clip(d.y_pred.to_numpy(float), 0, 125)
    r0, l0 = score(p0, y)
    imp = []
    for t in THRS:
        for dl in DLS:
            for dh in DHS:
                q = p0.copy(); m = q < t; q[m] += dl; q[~m] += dh
                r, l = score(q, y)
                if l < l0: imp.append((t, dl, dh, l - l0))
    tot = len(THRS)*len(DLS)*len(DHS)
    print(f'{cell:12s} improving params: {len(imp):3d}/{tot} ({100*len(imp)/tot:.0f}%)  best dLate={min(x[3] for x in imp):+.3f}')
    if imp:
        ts = sorted({x[0] for x in imp}); print(f'             thr range with any improvement: {ts[0]}-{ts[-1]}')
print()
print('fixed rule (40,-3,-2) improving in: see above cell lines')
