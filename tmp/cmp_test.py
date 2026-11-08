import glob
import numpy as np, pandas as pd
from pathlib import Path
base = Path('/home/user/rulb/experiments/TESTCACHE')
inn = Path('/home/user/rulb/experiments/TESTCACHE_IN')
def rmse(p):
    d = pd.read_csv(p)
    return float(np.sqrt(np.mean((d.y_pred - d.y_true)**2)))
bs = [rmse(p) for p in sorted(base.glob('FD003_s101_f*.csv'))]
ins = [rmse(p) for p in sorted(inn.glob('FD003_s101_f*.csv'))]
print('baseline folds:', [round(x,3) for x in bs])
print('instnorm folds:', [round(x,3) for x in ins])
print('baseline mean %.4f  instnorm mean %.4f  delta %+.2f%%' % (np.mean(bs), np.mean(ins), 100*(np.mean(ins)/np.mean(bs)-1)))
# ensembles
ds = [pd.read_csv(p).sort_values('unit').reset_index(drop=True) for p in sorted(base.glob('FD003_s101_f*.csv'))]
y = ds[0].y_true.to_numpy(); P = np.stack([d.y_pred.to_numpy() for d in ds])
print('baseline ensemble %.4f' % np.sqrt(np.mean((P.mean(0)-y)**2)))
ds2 = [pd.read_csv(p).sort_values('unit').reset_index(drop=True) for p in sorted(inn.glob('FD003_s101_f*.csv'))]
P2 = np.stack([d.y_pred.to_numpy() for d in ds2])
print('instnorm ensemble %.4f' % np.sqrt(np.mean((P2.mean(0)-y)**2)))