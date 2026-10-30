"""Corrected validation: MDE = 2.8*SE (SE = sd/sqrt(n) in RMSE units)."""
import glob
import numpy as np, pandas as pd
def per_engine(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float)**2
    return d.groupby('unit_id').sq.mean()
a, b = per_engine('M1_global'), per_engine('M6_tfA')
j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
d = (j.b - j.a).to_numpy()
rmse = float(np.sqrt(j.a.mean())); sd = float(d.std(ddof=1))
print(f'sd_diff={sd:.1f} rmse={rmse:.2f}')
print(' n     formula     simulated     ratio')
rng = np.random.default_rng(20261022)
for n in (52, 130, 260):
    sims = []
    for _ in range(2000):
        t = rng.integers(0, len(d), n)
        se_rmse = d[t].std(ddof=1)/np.sqrt(n)/(2*rmse)
        sims.append(2.8*se_rmse)          # MDE, single factor
    sim = float(np.mean(sims))
    formula = 2.8*(sd/np.sqrt(n))/(2*rmse)
    print(f'{n:4d}  {formula:8.2f}  {sim:11.2f}  {sim/formula:7.2f}')
