import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_b.eval.metrics_v2 import engine_mean_rmse, pooled_rmse

out = {'backbone': 'B0', 'split': 'val', 'cuts': [0.4, 0.55, 0.7, 0.85, 0.95],
       'estimators': ['engine_mean_rmse', 'pooled_rmse'], 'subsets': {}}
for subset in ['FD001', 'FD002', 'FD003', 'FD004']:
    files = sorted(glob.glob(f'experiments/hc/cata_gate/preds/{subset}_s*.csv'))
    if not files:
        continue
    rows = []
    for f in files:
        p = pd.read_csv(f)
        seed = int(f.split('_s')[-1].replace('.csv', ''))
        rows.append(dict(
            seed=seed,
            n_engines=int(p.engine.nunique()),
            n_rows=int(len(p)),
            k1_engine_mean=engine_mean_rmse(p.y_cap, p.pred_k1, p.engine),
            k1_pooled=pooled_rmse(p.y_cap, p.pred_k1),
            gate_engine_mean=engine_mean_rmse(p.y_cap, p.pred_gate, p.engine),
            gate_pooled=pooled_rmse(p.y_cap, p.pred_gate),
            identity_err=float(pooled_rmse(p.y_cap, p.pred_k1) ** 2
                               - engine_mean_rmse(p.y_cap, p.pred_k1, p.engine) ** 2
                               - np.var([np.sqrt(np.mean((g.pred_k1 - g.y_cap) ** 2))
                                         for _, g in p.groupby('engine')], ddof=0)),
        ))
    out['subsets'][subset] = dict(
        seeds=[r['seed'] for r in rows],
        n_engines=rows[0]['n_engines'], n_rows=rows[0]['n_rows'],
        per_seed=rows,
        k1_engine_mean_mean=float(np.mean([r['k1_engine_mean'] for r in rows])),
        k1_pooled_mean=float(np.mean([r['k1_pooled'] for r in rows])),
        gate_engine_mean_mean=float(np.mean([r['gate_engine_mean'] for r in rows])),
        gate_pooled_mean=float(np.mean([r['gate_pooled'] for r in rows])),
    )
Path('results').mkdir(exist_ok=True)
Path('results/SHARED_R0.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'per_seed'}
                  for k, v in out['subsets'].items()}, indent=1))
