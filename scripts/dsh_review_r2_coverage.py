import sys, warnings, json
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from dataset import load_cmapss_train

pred = pd.read_csv('results/dsh/review_r2/C_predictions.csv')
pred = pred[(pred.view=='v1') & (pred.model=='ridge')].copy()
print('predictions used:', len(pred))
rows = []
for ds in ('FD001','FD002'):
    df = load_cmapss_train(Path('data/CMAPSS'), ds)
    sett = ['op_1','op_2','op_3']
    for u, g in df.groupby('unit'):
        g = g.sort_values('cycle').reset_index(drop=True)
        for c in (60,100,140):
            if len(g) < c: continue
            w = g.iloc[c-50:c]
            rows.append(dict(dataset=ds, engine=int(u), cutoff=c,
                             disp=float(w[sett].std().mean()),
                             nid=int(w[sett[0]].nunique())))
cov = pd.DataFrame(rows)
m = pred.merge(cov, on=['dataset','engine','cutoff'], how='inner')
print('merged:', len(m))
m['err'] = m.y_pred - m.y_true
q1, q2 = m.disp.quantile([1/3, 2/3])
def band(x): return 'low_disp' if x <= q1 else ('mid' if x <= q2 else 'high_disp')
m['band'] = m.disp.map(band)
out = {}
print(f"{'band':10s} {'n':>4s} {'disp':>7s} {'RMSE':>7s} {'late':>6s}")
for b in ('low_disp','mid','high_disp'):
    s = m[m.band==b]
    r = float(np.sqrt((s.err**2).mean())); l = float((s.err>0).mean())
    out[b] = dict(n=int(len(s)), disp=float(s.disp.mean()), rmse=r, late=l)
    print(f'{b:10s} {len(s):4d} {s.disp.mean():7.3f} {r:7.3f} {l:6.3f}')
mono = out['low_disp']['rmse'] > out['mid']['rmse'] > out['high_disp']['rmse']
print()
print('monotone decreasing RMSE with dispersion:', mono)
out['monotone'] = bool(mono)
json.dump(out, open('results/dsh/review_r2/F_coverage.json','w'), indent=2)
