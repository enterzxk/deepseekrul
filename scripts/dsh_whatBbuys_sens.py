import glob, itertools
import numpy as np, pandas as pd
def per_engine(cell):
    frames = []
    for f in glob.glob(f'experiments/order03/{cell}/FD002/*/matched_cutoff_predictions.csv'):
        d = pd.read_csv(f); d['sq'] = d.signed_error.astype(float)**2
        frames.append(d.groupby('unit_id').sq.mean())
    return pd.concat(frames, axis=1).mean(axis=1)
cells = ['M1_global','M5_globalgate','M6_tfA','M4_racdr','M7_rawback','M4b_popgate','M0_raw_tf']
pe = {c: per_engine(c) for c in cells}
pairs = []
for a, b in itertools.combinations(cells, 2):
    j = pd.concat([pe[a].rename('a'), pe[b].rename('b')], axis=1).dropna()
    x, y = j.a.to_numpy(), j.b.to_numpy()
    pairs.append((np.sqrt(x.mean()) - np.sqrt(y.mean()), (y-x).std(ddof=1), np.sqrt(x.mean())))
n = 509
print('SD scale  decidable/21')
for s in (0.78, 1.00, 1.23):
    cnt = 0
    for diff, sd, rmse in pairs:
        mde = 2.8 * (sd*s) / (np.sqrt(n) * 2 * rmse)
        if abs(diff) > mde: cnt += 1
    print(f'  x{s:4.2f}      {cnt:2d}/21')
print()
print('note: 0.78 and 1.23 are the round-155 observed bounds on how pooling changes the SD')
