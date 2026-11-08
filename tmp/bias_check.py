import glob
import numpy as np, pandas as pd
from pathlib import Path
V = Path('/home/user/rulb/experiments')
C = Path('/home/user/rulb/experiments/TESTCACHE')
for sub in ['FD001','FD002','FD003','FD004']:
    vals = []
    for f in sorted(V.glob('M_%s_s*_f*/%s/dspt_s*.csv' % (sub, sub))):
        d = pd.read_csv(f)
        vals.append(float((d.y_pred - d.y_true).mean()))
    tst = []
    for f in sorted(C.glob('%s_s*_f*.csv' % sub)):
        d = pd.read_csv(f)
        tst.append(float((d.y_pred - d.y_true).mean()))
    v = np.array(vals); t = np.array(tst)
    print('%-6s val bias: mean %+.3f sd %.3f range [%+.2f, %+.2f]  |  test bias: mean %+.3f sd %.3f range [%+.2f, %+.2f]'
          % (sub, v.mean(), v.std(ddof=1), v.min(), v.max(), t.mean(), t.std(ddof=1), t.min(), t.max()))