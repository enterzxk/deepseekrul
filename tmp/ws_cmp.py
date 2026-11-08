import glob
import numpy as np, pandas as pd
from pathlib import Path
b = Path('/home/user/rulb/experiments/TESTCACHE')
w = Path('/home/user/rulb/experiments/TESTCACHE_WS')
for tag, pat in [('baseline', b), ('wshift', w)]:
    fs = sorted(pat.glob('FD003_s101_f*.csv'))
    ds = [pd.read_csv(f).sort_values('unit').reset_index(drop=True) for f in fs]
    y = ds[0].y_true.to_numpy()
    P = np.stack([d.y_pred.to_numpy() for d in ds])
    bias = float((P.mean(0) - y).mean())
    per = [float((d.y_pred.to_numpy() - y).mean()) for d in ds]
    print('%-9s n=%d  mean RMSE %.4f  ensemble %.4f  bias %+.3f  per-fold bias %s'
          % (tag, len(fs), np.mean([np.sqrt(np.mean((p-y)**2)) for p in P]), np.sqrt(np.mean((P.mean(0)-y)**2)), bias, [round(x,2) for x in per]))