"""Validate the MDE arithmetic by simulation on real per-engine data."""
import glob
import numpy as np, pandas as pd
def per_engine(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float)**2
    return d.groupby('unit_id').sq.mean()
a, b = per_engine('M1_global'), per_engine('M6_tfA')
j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
d = (j.b - j.a).to_numpy()          # per-engine MSE difference
rmse = float(np.sqrt(j.a.mean()))
sd = d.std(ddof=1)
print(f'n_engines={len(d)} sd_diff={sd:.1f} rmse={rmse:.2f}')
print()
print(' n   formula MDE   simulated CI half-width x2.8   ratio')
rng = np.random.default_rng(20261022)
for n in (52, 130, 260):
    halves = []
    for _ in range(2000):
        t = rng.integers(0, len(d), n)
        se = d[t].std(ddof=1)/np.sqrt(n)
        halves.append(1.96*se/(2*rmse))
    sim = float(np.mean(halves))*2.8
    formula = 2.8*(sd/np.sqrt(n))/(2*rmse)
    print(f'{n:4d}  {formula:10.2f}   {sim:26.2f}   {sim/formula:5.2f}')
