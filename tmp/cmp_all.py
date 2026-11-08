import glob
import numpy as np, pandas as pd
from pathlib import Path
base = Path('/home/user/rulb/experiments/TESTCACHE')
inn = Path('/home/user/rulb/experiments/TESTCACHE_IN')
def rmse(p):
    d = pd.read_csv(p)
    return float(np.sqrt(np.mean((d.y_pred - d.y_true)**2)))
for sub in ['FD001','FD002','FD003','FD004']:
    bs = sorted(base.glob('%s_s101_f*.csv' % sub))
    ins = sorted(inn.glob('%s_s101_f*.csv' % sub))
    if not bs or not ins:
        print('%s: base=%d inorm=%d (incomplete)' % (sub, len(bs), len(ins))); continue
    b = [rmse(p) for p in bs]; i = [rmse(p) for p in ins]
    def ens(paths):
        ds = [pd.read_csv(p).sort_values('unit').reset_index(drop=True) for p in paths]
        y = ds[0].y_true.to_numpy()
        P = np.stack([d.y_pred.to_numpy() for d in ds])
        return float(np.sqrt(np.mean((P.mean(0)-y)**2)))
    print('%-6s single base %.3f  inorm %.3f  (%+.2f%%)   ens base %.3f  inorm %.3f  (%+.2f%%)'
          % (sub, np.mean(b), np.mean(i), 100*(np.mean(i)/np.mean(b)-1), ens(bs), ens(ins), 100*(ens(ins)/ens(bs)-1)))