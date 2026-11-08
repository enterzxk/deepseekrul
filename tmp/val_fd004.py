import glob, numpy as np, pandas as pd
fs = sorted(glob.glob('/home/user/rulb/experiments/TESTCACHE/FD004_*.csv'))
print('FD004 cached:', len(fs))
for f in fs:
    d = pd.read_csv(f)
    e = d.y_pred - d.y_true
    print('%s n=%d  RMSE %.3f  bias %+.2f  max|e| %.1f  pred range [%.1f, %.1f]'
          % (f.split('/')[-1], len(d), np.sqrt(np.mean(e**2)), e.mean(), np.abs(e).max(), d.y_pred.min(), d.y_pred.max()))
if fs:
    ds = [pd.read_csv(f).sort_values('unit').reset_index(drop=True) for f in fs]
    y = ds[0].y_true.to_numpy()
    P = np.stack([d.y_pred.to_numpy() for d in ds])
    print('true range [%.1f, %.1f]  mean %.1f' % (y.min(), y.max(), y.mean()))
    print('ENSEMBLE %.3f' % np.sqrt(np.mean((P.mean(0)-y)**2)))