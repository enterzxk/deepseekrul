import glob, os
import numpy as np, pandas as pd
for f in sorted(glob.glob('/home/user/rulb/experiments/D1_FD002_*/FD002/dspt_s101.csv')):
    d = pd.read_csv(f)
    pooled = float(np.sqrt(np.mean((d.y_pred - d.y_true) ** 2)))
    per_eng = d.groupby('unit').apply(lambda g: np.sqrt(np.mean((g.y_pred - g.y_true) ** 2)), include_groups=False)
    eq = float(per_eng.mean())
    # engine-equal MSE then sqrt (project convention)
    mse_eq = float(np.sqrt(d.groupby('unit').apply(lambda g: np.mean((g.y_pred - g.y_true) ** 2), include_groups=False).mean()))
    tag = os.path.basename(os.path.dirname(os.path.dirname(f)))
    print('%-28s n_win=%6d n_eng=%3d  pooled %.4f  engine-equal-RMSE %.4f  sqrt(engine-equal-MSE) %.4f' % (tag, len(d), d.unit.nunique(), pooled, eq, mse_eq))