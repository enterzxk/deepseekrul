"""Is the paired-difference SD smaller than the per-engine SD? (corrects the MDE arithmetic)"""
import glob
import numpy as np, pandas as pd
def sq(root, tag):
    fs = glob.glob(f'experiments/{root}/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float) ** 2
    return d.groupby('unit_id').sq.mean()
a, b = sq('order03','M1_global'), sq('order03','M6_tfA')
j = pd.concat([a.rename('m1'), b.rename('m6')], axis=1).dropna()
print(f'n={len(j)}')
print(f'SD of per-engine MSE:  M1={j.m1.std():.1f}  M6={j.m6.std():.1f}')
print(f'SD of per-engine DIFF: {(j.m6-j.m1).std():.1f}')
print(f'corr of per-engine MSE: {j.m1.corr(j.m6):.3f}')
print()
sd_diff = float((j.m6-j.m1).std())
for n in (52, 124, 260):
    se = sd_diff/np.sqrt(n)
    # convert MSE-scale difference to RMSE-scale roughly: dRMSE ~ dMSE/(2*RMSE)
    rmse = np.sqrt(j.m1.mean())
    print(f'n={n:4d}: SE(MSE diff)={se:7.1f} -> SE(RMSE diff)~{se/(2*rmse):5.2f} -> MDE~{2.8*se/(2*rmse):5.2f} RMSE')
