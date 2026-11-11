"""Recompute historical table entries from the archived prediction CSVs."""
import glob
import os

import numpy as np
import pandas as pd

ROOT = 'paper_b/archive/legacy_2026_11/results'


def nasa(p, y):
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)))


rows = []
for f in sorted(glob.glob(os.path.join(ROOT, 'ABLTEST', '*.csv'))):
    name = os.path.basename(f).replace('.csv', '')
    subset, cfg = name.split('_', 1)
    d = pd.read_csv(f)
    p, y = d.y_pred.values, d.y_true.values
    rows.append(dict(subset=subset, cfg=cfg, n=len(d), rmse=float(np.sqrt(np.mean((p-y)**2))), nasa=nasa(p, y)))
new = pd.DataFrame(rows)
new.to_csv('paper_b/results/legacy_abltest_recomputed.csv', index=False)
print(new.pivot(index='cfg', columns='subset', values='rmse').round(3).to_string())
print()
t = pd.read_csv(os.path.join(ROOT, 'abl_test2.csv'))
print('table vs recomputed max |diff|:',
      float((t.set_index(['subset','cfg']).rmse - new.set_index(['subset','cfg']).rmse).abs().max()))
