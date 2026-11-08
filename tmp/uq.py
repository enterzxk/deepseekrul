"""D5b: uncertainty quantification on the official test set (deep-ensemble + CQR)."""
import glob, json
import numpy as np, pandas as pd
from pathlib import Path
C = Path('/home/user/rulb/experiments/TESTCACHE')
V = Path('/home/user/rulb/experiments')
ALPHA = [0.10, 0.05]

def cqr_scores(pred, lo, hi, y):
    return np.maximum(lo - y, y - hi)

rows = []
for subset in ['FD001','FD002','FD003','FD004']:
    fs = sorted(C.glob(f'{subset}_s*_f*.csv'))
    ds = [pd.read_csv(f).sort_values('unit').reset_index(drop=True) for f in fs]
    y = ds[0].y_true.to_numpy()
    P = np.stack([d.y_pred.to_numpy() for d in ds])
    mu = P.mean(0); sd = P.std(0, ddof=1)
    out = {'subset': subset, 'n': len(y), 'rmse': float(np.sqrt(np.mean((mu-y)**2)))}
    # (a) raw deep-ensemble sigma intervals
    for a in ALPHA:
        z = {0.10: 1.645, 0.05: 1.96}[a]
        lo, hi = mu - z*sd, mu + z*sd
        out['cov%.0f_raw' % (100*(1-a))] = float(np.mean((y >= lo) & (y <= hi)))
        out['wid%.0f_raw' % (100*(1-a))] = float(np.mean(hi - lo))
    # (b) CQR calibrated on the training-fold validation predictions
    vals = []
    for f in fs:
        tag = f.stem
        p = V / ('M_' + tag) / subset / ('dspt_s' + tag.split('_s')[1] + '.csv')
        if p.is_file(): vals.append(pd.read_csv(p))
    if vals:
        vy = np.concatenate([v.y_true.to_numpy() for v in vals])
        vp = np.concatenate([v.y_pred.to_numpy() for v in vals])
        # residual quantile as the conformal score (absolute residual)
        resid = np.abs(vp - vy)
        for a in ALPHA:
            q = float(np.quantile(resid, 1 - a))
            lo, hi = mu - q, mu + q
            out['cov%.0f_cqr' % (100*(1-a))] = float(np.mean((y >= lo) & (y <= hi)))
            out['wid%.0f_cqr' % (100*(1-a))] = float(np.mean(hi - lo))
    rows.append(out)

r = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(r.round(3).to_string(index=False))
r.to_csv('/home/user/rulb/RESULTS_UQ.csv', index=False)