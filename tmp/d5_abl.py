"""D5 ablation: long vs short vs dual vs slot vs repeat, from the D1 screening runs."""
import glob, os
import numpy as np, pandas as pd
rows = []
for d in sorted(glob.glob('/home/user/rulb/experiments/D1_*')):
    name = os.path.basename(d)
    parts = name.split('_')
    if len(parts) < 4: continue
    subset, mode, fold = parts[1], parts[2], parts[3]
    f = f'{d}/{subset}/dspt_s101.csv'
    if not os.path.isfile(f): continue
    df = pd.read_csv(f)
    e = df.y_pred - df.y_true
    g = pd.DataFrame({'u': df.unit, 'e2': e**2}).groupby('u').mean()
    rows.append(dict(subset=subset, mode=mode, fold=fold, rmse=float(np.sqrt(g.e2.mean()))))
r = pd.DataFrame(rows)
if r.empty:
    print('no D1 data')
else:
    piv = r.pivot_table(index=['subset','fold'], columns='mode', values='rmse')
    print(piv.round(3).to_string())
    print()
    print('mean over folds:')
    for subset in ['FD002','FD004']:
        s = r[r.subset == subset]
        if s.empty: continue
        m = s.groupby('mode').rmse.mean().sort_values()
        base = m.get('long', np.nan)
        line = '  %s: ' % subset
        for k, v in m.items():
            line += '%s %.3f (%+.1f%%)  ' % (k, v, 100*(v/base-1) if np.isfinite(base) else np.nan)
        print(line)
    n_slot = (r['mode']=='slot').sum()
    print()
    print('slot runs found:', n_slot)