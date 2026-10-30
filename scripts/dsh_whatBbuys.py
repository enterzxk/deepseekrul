"""Power projection: which of the 21 pairs would become decidable at n=509?"""
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
rows = []
for a, b in itertools.combinations(cells, 2):
    j = pd.concat([pe[a].rename('a'), pe[b].rename('b')], axis=1).dropna()
    x, y = j.a.to_numpy(), j.b.to_numpy()
    diff = np.sqrt(x.mean()) - np.sqrt(y.mean())
    sd = (y - x).std(ddof=1)
    rmse = np.sqrt(x.mean())
    # SE of the RMSE difference ~ sd / (sqrt(n) * 2*rmse); MDE = 2.8*SE
    for n in (52, 509):
        se = sd / (np.sqrt(n) * 2 * rmse)
        mde = 2.8 * se
        if n == 509:
            rows.append((a, b, diff, sd, mde, abs(diff) > mde))
df = pd.DataFrame(rows, columns=['a','b','diff','sd','mde509','decidable'])
print(f'pairs={len(df)}  decidable at 509: {int(df.decidable.sum())}')
print(df[df.decidable].sort_values('diff', key=abs, ascending=False).to_string(index=False, float_format=lambda v: f'{v:8.2f}'))
print()
print('largest effects that would STILL be undecidable:')
nd = df[~df.decidable].reindex(df[~df.decidable].diff.abs().sort_values(ascending=False).index)
print(nd.head(5).to_string(index=False, float_format=lambda v: f'{v:8.2f}'))
