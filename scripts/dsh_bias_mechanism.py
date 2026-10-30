"""Does correction benefit track the model's existing bias?"""
import glob
import numpy as np, pandas as pd
def score(p, y):
    e = p - y
    return float(np.sqrt((e**2).mean())), float((e>0).mean())
cells = ['M1_global','M5_globalgate','M6_tfA','M4_racdr','M7_rawback','M4b_popgate','M0_raw_tf']
rows = []
for c in cells:
    fs = glob.glob(f'experiments/order03/{c}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: continue
    d = pd.concat([pd.read_csv(f) for f in fs])
    y = d.y_true.to_numpy(float); p = np.clip(d.y_pred.to_numpy(float), 0, 125)
    bias = float((p - y).mean())
    r0, l0 = score(p, y)
    q = p.copy(); m = q < 40; q[m] += -3; q[~m] += -2; q = np.clip(q, 0, 125)
    r1, l1 = score(q, y)
    rows.append((c, bias, r0, r1 - r0, l1 - l0))
t = pd.DataFrame(rows, columns=['cell','bias','rmse0','dRMSE','dLate'])
print(t.to_string(index=False, float_format=lambda v: f'{v:8.3f}'))
r = np.corrcoef(t.bias, t.dRMSE)[0,1]
print()
print(f'corr(bias, dRMSE) = {r:+.3f}   (expect negative: more bias -> more RMSE gain)')
print(f'corr(bias, |dLate|) = {np.corrcoef(t.bias, -t.dLate)[0,1]:+.3f}')
