"""Pareto frontier of (RMSE cost, late reduction); is the chosen rule efficient?"""
import glob
import numpy as np, pandas as pd
THRS = (30,40,50,60,70,80); DLS = (-12,-9,-6,-3); DHS = (0,-2,-4)
def score(p, y):
    e = np.clip(p,0,125) - y
    return float(np.sqrt((e**2).mean())), float((e>0).mean())
d = pd.concat([pd.read_csv(f) for f in glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')])
y = d.y_true.to_numpy(float); p0 = np.clip(d.y_pred.to_numpy(float), 0, 125)
r0, l0 = score(p0, y)
rows = []
for t in THRS:
    for dl in DLS:
        for dh in DHS:
            q = p0.copy(); m = q < t; q[m] += dl; q[~m] += dh
            r, l = score(q, y)
            rows.append((t, dl, dh, r - r0, l - l0))
df = pd.DataFrame(rows, columns=['thr','dl','dh','dR','dL'])
# Pareto: minimise dR and dL
front = []
for i, a in df.iterrows():
    dominated = ((df.dR <= a.dR) & (df.dL <= a.dL) & ((df.dR < a.dR) | (df.dL < a.dL))).any()
    if not dominated: front.append(a)
f = pd.DataFrame(front).sort_values('dL')
print(f'pareto-optimal configurations: {len(f)} of {len(df)}')
print(f.chr if False else f.to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
print()
chosen = df[(df.thr==40)&(df.dl==-3)&(df.dh==-2)].iloc[0]
print(f'chosen (40,-3,-2): dR={chosen.dR:+.3f} dL={chosen.dL:+.3f}  on frontier={any(abs(chosen.dR-x.dR)<1e-9 and abs(chosen.dL-x.dL)<1e-9 for _,x in f.iterrows())}')
