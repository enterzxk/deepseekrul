"""Is the padding-mask effect concentrated in short (heavily padded) test engines?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/user/rulb')
test = pd.read_csv(ROOT / 'data' / 'CMAPSS' / 'test_FD001.txt', sep=r'\s+', header=None)
lengths = test.groupby(0).size().rename('len').reset_index().rename(columns={0: 'unit'})
rows = []
for tag in ('ATTR_fixed', 'ATTR_unmasked'):
    d = pd.read_csv(ROOT / 'paper_b' / 'results' / 'raw' / tag / 'FD001' / 's101' / 'test_pred.csv')
    d = d.merge(lengths, on='unit')
    for name, sel in (('short(<70)', d.len < 70), ('full(>=70)', d.len >= 70)):
        e = d.y_pred[sel] - d.y_true[sel]
        rows.append(dict(tag=tag, group=name, n=int(sel.sum()),
                         rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean())))
    e = d.y_pred - d.y_true
    rows.append(dict(tag=tag, group='all', n=len(d), rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean())))
print(pd.DataFrame(rows).to_string(index=False))
