import numpy as np
from pathlib import Path
import sys
sys.path[:0] = ['/home/user/rulb/src', '/home/user/rulb']
from dataset import load_cmapss_train
ROOT = Path('/home/user/rulb')
CAP = 125.0
for sub in ['FD001','FD003']:
    raw = load_cmapss_train(ROOT/'data'/'CMAPSS', sub)
    lens = raw.groupby('unit').size().to_numpy()
    # p_train(y): empirical distribution of clipped targets over all (engine, t)
    tg = np.concatenate([np.clip(n - np.arange(n), 0, CAP) for n in lens])
    # p_test(y) under uniform truncation: P(RUL = y) proportional to #{engines with n > y}
    ys = np.arange(0, int(CAP) + 1)
    p_test = np.array([(lens > y).sum() for y in ys], float)
    p_test = np.minimum(p_test, (lens > CAP).sum())  # capped targets all map to CAP
    p_test = p_test / p_test.sum()
    p_tr = np.array([(np.round(tg) == y).sum() for y in ys], float)
    p_tr = p_tr / p_tr.sum()
    w = np.where(p_tr > 1e-9, p_test / np.maximum(p_tr, 1e-9), 0.0)
    w = w / (p_tr * w).sum()  # normalise so E_train[w] = 1
    print('%-6s engines %d  mean len %.0f' % (sub, len(lens), lens.mean()))
    print('   p_train mean RUL %.1f   p_test mean RUL %.1f' % ((ys*p_tr).sum(), (ys*p_test).sum()))
    for y in [25, 50, 75, 100, 125]:
        print('   y=%3d  p_train %.4f  p_test %.4f  w %.3f' % (y, p_tr[y], p_test[y], w[y]))
    print('   weight range [%.3f, %.3f]  effective sample size %.1f%%' % (w.min(), w.max(), 100*(p_tr*w).sum()**2/((p_tr*w**2).sum())))