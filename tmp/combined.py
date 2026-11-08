"""Combined ensemble: 15 baseline models + 5 instance-normalised models per subset."""
import glob
import numpy as np, pandas as pd
from pathlib import Path
base = Path('/home/user/rulb/experiments/TESTCACHE')
inn = Path('/home/user/rulb/experiments/TESTCACHE_IN')
def load(paths):
    ds = [pd.read_csv(p).sort_values('unit').reset_index(drop=True) for p in paths]
    return ds[0].y_true.to_numpy(), np.stack([d.y_pred.to_numpy() for d in ds])
rows = []
for sub in ['FD001','FD002','FD003','FD004']:
    bs = sorted(base.glob('%s_s*_f*.csv' % sub))
    ins = sorted(inn.glob('%s_s*_f*.csv' % sub))
    if not bs:
        continue
    y, PB = load(bs)
    r = {'subset': sub, 'n_base': PB.shape[0], 'n_in': 0,
         'rmse_base': float(np.sqrt(np.mean((PB.mean(0)-y)**2)))}
    if ins:
        _, PI = load(ins)
        r['n_in'] = PI.shape[0]
        both = np.concatenate([PB, PI], axis=0)
        r['rmse_combined'] = float(np.sqrt(np.mean((both.mean(0)-y)**2)))
        r['rmse_in_only'] = float(np.sqrt(np.mean((PI.mean(0)-y)**2)))
    rows.append(r)
df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(df.round(4).to_string(index=False))
df.to_csv('/home/user/rulb/RESULTS_COMBINED.csv', index=False)