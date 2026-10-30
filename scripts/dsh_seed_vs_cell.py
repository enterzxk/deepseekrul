"""Is the M1 vs M4 difference larger than seed-to-seed variation?"""
import glob, itertools
import numpy as np, pandas as pd
def seed_preds(tag):
    out = {}
    for f in glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv'):
        seed = f.split('/')[-2].split('_s')[-1]
        d = pd.read_csv(f)
        out[seed] = d.groupby('unit_id').y_pred.mean()
    return out
for tag in ('M1_global','M4_racdr'):
    P = seed_preds(tag)
    ks = sorted(P)
    print(f'{tag}: seeds={ks}')
    for a, b in itertools.combinations(ks, 2):
        j = pd.concat([P[a].rename('x'), P[b].rename('y')], axis=1).dropna()
        print(f'   seed {a} vs {b}: corr={j.x.corr(j.y):+.4f}  RMSE_agree={np.sqrt(((j.x-j.y)**2).mean()):.2f}')
m1 = seed_preds('M1_global'); m4 = seed_preds('M4_racdr')
print()
print('cross-cell (same seed):')
for s in sorted(set(m1) & set(m4)):
    j = pd.concat([m1[s].rename('x'), m4[s].rename('y')], axis=1).dropna()
    print(f'   seed {s}: corr={j.x.corr(j.y):+.4f}')
