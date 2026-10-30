"""Cross-validated downward calibration: estimate the shift on held-out engines."""
import glob
import numpy as np, pandas as pd
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
g = d.groupby('unit_id').agg(y=('y_true','mean'), p=('y_pred','mean'))
y, p = g.y.to_numpy(float), g.p.to_numpy(float)
rng = np.random.default_rng(20261022)
res = {0: [], 2: [], 4: []}
for _ in range(500):
    idx = rng.permutation(len(y)); h = len(y)//2
    cal, ev = idx[:h], idx[h:]
    bias = float(np.mean(p[cal] - y[cal]))          # estimated on calibration engines
    for s in res:
        q = p[ev] - bias + (-s if s else 0)
        e = q - y[ev]
        res[s].append((np.sqrt((e**2).mean()), (e>0).mean(), (e>10).sum()/len(ev), (e<-10).mean()))
print('CV: shift applied = -bias (+ extra)')
print(' extra  rmse   late  severe  early>10')
for s, v in res.items():
    a = np.array(v).mean(0)
    print(f'{s:6d} {a[0]:6.2f} {a[1]:6.3f} {a[2]:6.3f} {a[3]:7.3f}')
print()
print('mean out-of-fold bias correction:', round(float(np.mean(p-y)),2), '(in-sample would be optimistic)')
