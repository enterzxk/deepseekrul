import numpy as np, pandas as pd
from pathlib import Path
import sys
sys.path[:0] = ['/home/user/rulb/src', '/home/user/rulb']
from dataset import load_cmapss_train, load_cmapss_files
ROOT = Path('/home/user/rulb')
for sub in ['FD001','FD002','FD003','FD004']:
    raw = load_cmapss_train(ROOT/'data'/'CMAPSS', sub)
    # training targets: clip(n-t,0,125) over all cycles
    tg = []
    for u, g in raw.groupby('unit'):
        n = len(g)
        tg.append(np.clip(n - np.arange(n), 0, 125))
    tg = np.concatenate(tg)
    _, tf, rul = load_cmapss_files(ROOT/'data'/'CMAPSS', sub)
    te = np.minimum(np.asarray(rul, np.float64), 125.0)
    print('%-6s train targets: mean %5.1f  median %5.1f  frac<=50 %.3f  frac>=100 %.3f  |  test RUL: mean %5.1f  median %5.1f  frac<=50 %.3f  frac>=100 %.3f'
          % (sub, tg.mean(), np.median(tg), (tg<=50).mean(), (tg>=100).mean(), te.mean(), np.median(te), (te<=50).mean(), (te>=100).mean()))