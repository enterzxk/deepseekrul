"""How much of the paired-difference variance is seed noise (fixable by more seeds)?"""
import glob
import numpy as np, pandas as pd
def per_seed(root, tag):
    fs = glob.glob(f'experiments/{root}/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    rows = []
    for f in fs:
        seed = int(f.split('_s')[-1].split('/')[0])
        d = pd.read_csv(f)
        d['sq'] = d.signed_error.astype(float) ** 2
        g = d.groupby('unit_id').sq.mean().rename(seed)
        rows.append(g)
    return pd.concat(rows, axis=1)
a = per_seed('order03','M1_global'); b = per_seed('order03','M6_tfA')
if a is None or b is None: print('missing'); raise SystemExit
seeds = sorted(set(a.columns) & set(b.columns))
print('seeds:', seeds)
diffs = a[seeds].to_numpy() - b[seeds].to_numpy()   # per engine x seed
cell_mean = diffs.mean(axis=1)
seed_var = float(np.var(diffs - cell_mean[:,None], ddof=1))   # within-engine across-seed
tot_var = float(np.var(cell_mean, ddof=1))
print(f'SD of seed-averaged diff      : {np.sqrt(tot_var):7.1f}')
print(f'seed noise (within-engine) SD : {np.sqrt(seed_var):7.1f}')
print(f'share of variance from seeds  : {100*seed_var/(tot_var+seed_var/len(seeds)):5.1f}%')
print()
rmse = np.sqrt(a[seeds].to_numpy().mean())
for k in (3, 5, 10, 999):
    v = tot_var + seed_var / (k if k < 999 else 1e9)
    se = np.sqrt(v/52) / (2*rmse)      # RMSE-scale SE at n=52 engines
    se260 = se*np.sqrt(52/260)
    print(f'seeds={k if k<999 else "inf":>4}: MDE(n=52)={2.8*se:5.2f}  MDE(n=260)={2.8*se260:5.2f} RMSE')
