"""One FIXED parameter set applied to all configurations (no per-case tuning)."""
import glob
import numpy as np, pandas as pd
THR, DL, DH = 40, -3, -2
def apply(p):
    q = np.clip(p,0,125).copy(); m = q < THR; q[m] += DL; q[~m] += DH
    return np.clip(q, 0, 125)
def score(p, y):
    e = p - y
    return float(np.sqrt((e**2).mean())), float((e>0).mean()), float((e>30).mean())
cases = [('M1_global','FD002','matched_cutoff_predictions.csv'),
         ('M1_global','FD001','matched_cutoff_predictions.csv'),
         ('M6_tfA','FD002','matched_cutoff_predictions.csv'),
         ('M0_raw_tf','FD002','matched_cutoff_predictions.csv'),
         ('M1_global','FD002','protocol_c_predictions.csv')]
print(f"{'config':34s} {'n':>5s} {'dRMSE':>7s} {'dLate':>7s} {'dSevere':>8s}")
for cell, sub, kind in cases:
    fs = glob.glob(f'experiments/order03/{cell}/{sub}/*/{kind}')
    if not fs: continue
    d = pd.concat([pd.read_csv(f) for f in fs])
    y = d.y_true.to_numpy(float); p = np.clip(d.y_pred.to_numpy(float), 0, 125)
    r0, l0, s0 = score(p, y); r1, l1, s1 = score(apply(p), y)
    print(f'{cell+"/"+sub+"/"+kind.split("_")[0]:34s} {len(y):5d} {r1-r0:+7.3f} {l1-l0:+7.3f} {s1-s0:+8.3f}')
