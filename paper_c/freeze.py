"""Select and freeze B_min from Phase 0 results."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

rows = json.loads(Path('results/paper_c/p0_selection.json').read_text())
g = defaultdict(list)
for r in rows:
    g[(r['subset'], r['model'], r['L'])].append(r['engine_mean_rmse'])
configs = sorted({(m, L) for _, m, L in g})
table = []
for m, L in configs:
    vals = {s: np.asarray(g.get((s, m, L), []), float) for s in ['FD001', 'FD004']}
    if not all(len(v) for v in vals.values()):
        continue
    mean = float(np.mean([vals[s].mean() for s in vals]))
    table.append(dict(model=m, L=L,
                      fd001=float(vals['FD001'].mean()), fd001_sd=float(vals['FD001'].std()),
                      fd004=float(vals['FD004'].mean()), fd004_sd=float(vals['FD004'].std()),
                      mean=mean))
best = min(table, key=lambda r: (r['mean'], r['fd001_sd'] + r['fd004_sd']))
out = dict(
    backbone=best['model'], L=best['L'], use_cycle=False,
    backbone_config=dict(hidden=64, layers=2, n_params=51073),
    cap=dict(model='gru', hidden=64, layers=3, L=50, n_params=76033),
    sensors=[2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
    cuts=[0.25, 0.40, 0.55, 0.70, 0.85, 0.95], folds=5, split_seed=2026,
    selection=table, selected=best,
)
Path('configs').mkdir(exist_ok=True)
Path('configs/backbone_freeze.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
