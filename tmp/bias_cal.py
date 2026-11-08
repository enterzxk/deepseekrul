"""Post-hoc bias calibration: estimate the mean residual on held-out folds, subtract it on test."""
import glob, os
import numpy as np, pandas as pd
from pathlib import Path
C = Path('/home/user/rulb/experiments/TESTCACHE')
V = Path('/home/user/rulb/experiments')
rows = []
for subset in ['FD001','FD002','FD003','FD004']:
    fs = sorted(C.glob(f'{subset}_s*_f*.csv'))
    ds = [pd.read_csv(f).sort_values('unit').reset_index(drop=True) for f in fs]
    y = ds[0].y_true.to_numpy()
    P = np.stack([d.y_pred.to_numpy() for d in ds])
    biases = []
    for f in fs:
        tag = f.stem; seed = tag.split('_s')[1].split('_')[0]
        p = V / ('M_' + tag) / subset / ('dspt_s' + seed + '.csv')
        if not p.is_file(): continue
        d = pd.read_csv(p)
        biases.append(float((d.y_pred - d.y_true).mean()))
    b = float(np.mean(biases)) if biases else 0.0
    # per-model calibration, then mean of calibrated RMSEs
    per = [np.sqrt(np.mean(((P[i] - b) - y)**2)) for i in range(len(P))]
    ens = np.sqrt(np.mean(((P.mean(0) - b) - y)**2))
    raw = [np.sqrt(np.mean((P[i] - y)**2)) for i in range(len(P))]
    rows.append(dict(subset=subset, n=len(P), val_bias=round(b,3),
                     single_raw=round(float(np.mean(raw)),4), single_cal=round(float(np.mean(per)),4),
                     ens_raw=round(float(np.sqrt(np.mean((P.mean(0)-y)**2))),4), ens_cal=round(float(ens),4)))
r = pd.DataFrame(rows)
print(r.to_string(index=False))