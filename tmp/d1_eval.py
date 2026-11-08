"""D1 gate analysis: dual vs best single stream, plus the P1 repeat-stream control."""
import glob, json, os
import numpy as np, pandas as pd
from pathlib import Path
H = Path('/home/user/rulb/experiments')

def rmse(p, y): return float(np.sqrt(np.mean((p - y) ** 2)))

rows = []
for d in sorted(glob.glob(str(H / 'D1_*'))):
    name = os.path.basename(d)
    parts = name.split('_')
    if len(parts) < 5: continue
    subset, mode, fold = parts[1], parts[2], parts[3]
    f = Path(d) / subset / 'dspt_s101.csv'
    if not f.is_file(): continue
    df = pd.read_csv(f)
    rows.append(dict(subset=subset, mode=mode, fold=fold, rmse=rmse(df.y_pred, df.y_true), n=len(df)))

if not rows:
    print('no results yet')
else:
    r = pd.DataFrame(rows)
    piv = r.pivot_table(index=['subset', 'fold'], columns='mode', values='rmse')
    print(piv.round(4).to_string())
    print()
    for subset in r.subset.unique():
        s = r[r.subset == subset]
        means = s.groupby('mode').rmse.mean()
        if not {'dual', 'short', 'long'} <= set(means.index): continue
        best_single = min(means['short'], means['long'])
        best_name = 'short' if means['short'] <= means['long'] else 'long'
        gain = 100 * (1 - means['dual'] / best_single)
        rep = means.get('repeat', np.nan)
        rep_gain = 100 * (1 - rep / best_single) if np.isfinite(rep) else np.nan
        need = 1.5 if subset == 'FD002' else 1.0
        print('%s: dual %.4f  best single (%s) %.4f  gain %+.2f%%  (need >= %.1f%%)  repeat %+.2f%%  -> %s'
              % (subset, means['dual'], best_name, best_single, gain, need, rep_gain,
                 'PASS' if gain >= need and (not np.isfinite(rep_gain) or rep_gain < 0.5) else 'FAIL'))