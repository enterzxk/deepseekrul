"""Export cut-wise and Phase-4 table data to JSON (run on lab3090)."""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from paper_b.eval.metrics_v2 import engine_mean_rmse

SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
FRACS = [0.4, 0.55, 0.7, 0.85, 0.95]
cut = {}
for s in SUBSETS:
    df = pd.read_csv(f'results/ltagv2_p1_{s}.preds.csv')
    df['rank'] = df.groupby('engine')['cut'].rank(method='dense').astype(int) - 1
    rows = []
    for r in range(5):
        sub = df[df['rank'] == r]
        rows.append(dict(frac=FRACS[r],
                         k1=engine_mean_rmse(sub.y, sub.p_k1, sub.engine),
                         ltag=engine_mean_rmse(sub.y, sub.pred_ltag, sub.engine),
                         calib=engine_mean_rmse(sub.y, sub.pred_calib, sub.engine),
                         ls=engine_mean_rmse(sub.y, sub.pred_ls, sub.engine)))
    cut[s] = rows
Path('results/cutwise.json').write_text(json.dumps(cut, indent=1))

g = defaultdict(lambda: defaultdict(list))
for p in sorted(glob.glob('experiments/phase4/logs/*.log')):
    m = re.match(r'(FD\d{3})_(\w+)_onset([\d.]+)_s(\d+)\.log', Path(p).name)
    if not m:
        continue
    text = Path(p).read_text(errors='replace')
    vals = {}
    for arm in ('R3', 'R5b', 'R5m'):
        mm = re.search(rf'{arm} [\d.]+ rec ([\-\d.]+|n/a)', text)
        if mm and mm.group(1) != 'n/a':
            vals[arm] = float(mm.group(1))
    if 'R3' not in vals:
        continue
    key = (m.group(1), m.group(2), float(m.group(3)))
    for k, v in vals.items():
        g[key][k].append(v)
out = {}
for k, v in g.items():
    out[f'{k[0]}|{k[1]}|{k[2]}'] = {a: float(np.mean(v[a])) for a in v}
Path('results/phase4.json').write_text(json.dumps(out, indent=1))
print('wrote results/cutwise.json and results/phase4.json')
