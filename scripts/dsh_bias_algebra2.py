import glob
import numpy as np, pandas as pd
cells = ['M1_global','M5_globalgate','M6_tfA','M4_racdr','M7_rawback','M4b_popgate','M0_raw_tf']
obs, alg, biases = [], [], []
print(f"{'cell':14s} {'bias':>7s} {'shift':>7s} {'dMSE_obs':>9s} {'dMSE_alg':>9s}")
for c in cells:
    fs = glob.glob(f'experiments/order03/{c}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: continue
    d = pd.concat([pd.read_csv(f) for f in fs])
    y = d.y_true.to_numpy(float); p = np.clip(d.y_pred.to_numpy(float), 0, 125)
    q = p.copy(); m = q < 40; q[m] += -3; q[~m] += -2; q = np.clip(q, 0, 125)
    e0, e1 = p - y, q - y
    s = float((q - p).mean()); b = float(e0.mean())
    dmse_obs = float((e1**2).mean() - (e0**2).mean())
    dmse_alg = s*s + 2*s*b          # correct sign
    obs.append(dmse_obs); alg.append(dmse_alg); biases.append(b)
    print(f'{c:14s} {b:7.3f} {s:7.2f} {dmse_obs:9.2f} {dmse_alg:9.2f}')
obs, alg, biases = np.array(obs), np.array(alg), np.array(biases)
print()
print(f'corr(alg, obs) = {np.corrcoef(alg, obs)[0,1]:+.3f}')
print(f'corr(bias, obs) = {np.corrcoef(biases, obs)[0,1]:+.3f}')
resid = obs - alg
print(f'residual (obs - alg): mean {resid.mean():+.2f} sd {resid.std():.2f} range [{resid.min():+.2f}, {resid.max():+.2f}]')
