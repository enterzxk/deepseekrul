"""Is the bias correlation algebraic rather than empirical?"""
import glob
import numpy as np, pandas as pd
def score(p, y):
    e = p - y
    return float(np.sqrt((e**2).mean())), float(e.mean()), float((e**2).mean())
cells = ['M1_global','M5_globalgate','M6_tfA','M4_racdr','M7_rawback','M4b_popgate','M0_raw_tf']
print(f"{'cell':14s} {'bias':>7s} {'dMSE_obs':>9s} {'dMSE_alg':>9s} {'dRMSE_obs':>10s} {'dRMSE_alg':>10s}")
for c in cells:
    fs = glob.glob(f'experiments/order03/{c}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: continue
    d = pd.concat([pd.read_csv(f) for f in fs])
    y = d.y_true.to_numpy(float); p = np.clip(d.y_pred.to_numpy(float), 0, 125)
    q = p.copy(); m = q < 40; q[m] += -3; q[~m] += -2; q = np.clip(q, 0, 125)
    r0, b0, mse0 = score(p, y)
    r1, b1, mse1 = score(q, y)
    shift = (q - p).mean()
    # algebra: dMSE ~ shift^2 - 2*shift*bias  (for a constant shift)
    dmse_alg = shift**2 - 2*shift*b0
    print(f'{c:14s} {b0:7.3f} {mse1-mse0:9.2f} {dmse_alg:9.2f} {r1-r0:10.3f} {dmse_alg/(2*r0):10.3f}')
print()
print('if dMSE_obs ~= dMSE_alg, the bias correlation is an ALGEBRAIC identity, not an empirical law')
