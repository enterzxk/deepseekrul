"""Per-engine analysis for the paper: error by lifetime quartile and concentration."""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path('/home/user/rulb/tim_20261105/full_20261105')

def load(subset, arm, seed=101):
    b = []
    for k in range(5):
        p = HERE / 'runs' / subset / str(k) / f'{arm}_{seed}.csv'
        if not p.is_file():
            return None
        b.append(pd.read_csv(p))
    return pd.concat(b, ignore_index=True)

rows = []
for subset in ['FD002', 'FD004']:
    d0 = load(subset, 'M0')
    if d0 is None:
        continue
    life = d0.groupby('unit').end_cycle.max()
    q = pd.qcut(life, 4, labels=['Q1 short', 'Q2', 'Q3', 'Q4 long'])
    for arm in ['M0', 'OPS', 'FULL']:
        d = load(subset, arm)
        if d is None:
            continue
        d['err'] = d.y_pred - d.y_true
        g = d.groupby('unit').agg(mse=('err', lambda x: float((x**2).mean())),
                                  bias=('err', 'mean'))
        g['rmse'] = np.sqrt(g.mse)
        for lab in q.cat.categories:
            u = life.index[q == lab]
            sub = g.loc[g.index.intersection(u)]
            rows.append(dict(subset=subset, arm=arm, stratum=lab, engines=len(sub),
                             rmse=float(np.sqrt(sub.mse.mean())), bias=float(sub.bias.mean())))
        top = g.sort_values('mse', ascending=False)
        n10 = max(1, int(0.1 * len(g)))
        rest = top.iloc[n10:]
        rows.append(dict(subset=subset, arm=arm, stratum='worst10%_removed',
                         engines=len(rest), rmse=float(np.sqrt(rest.mse.mean())),
                         bias=float(rest.bias.mean())))
frame = pd.DataFrame(rows)
frame.to_csv(HERE / 'PER_ENGINE.csv', index=False)
(HERE / 'paper_tables').mkdir(exist_ok=True)
print(frame.round(2).to_string(index=False))
