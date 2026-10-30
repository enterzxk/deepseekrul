import glob
import numpy as np, pandas as pd
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
g = d.groupby('unit_id').agg(y=('y_true','mean'), p=('y_pred','mean'))
g['e'] = g.p - g.y; g['ae'] = g.e.abs()
g = g.sort_values('ae', ascending=False)
print('worst 8 engines:')
print(g.head(8).round(1).to_string())
print()
for lo, hi in [(0,30),(30,60),(60,90),(90,126)]:
    s = g[(g.y>=lo)&(g.y<hi)]
    if len(s):
        print(f'true RUL {lo:3d}-{hi:3d}: n={len(s):2d} mean|e|={s.ae.mean():5.2f} max|e|={s.ae.max():5.1f} share_of_MSE={100*(s.e**2).sum()/(g.e**2).sum():5.1f}%')
