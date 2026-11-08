#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY - <<'PYEOF'
import pandas as pd, numpy as np
df = pd.read_csv('experiments/ida/prior/FD001_prior_cutoffs.csv')
for role, g in df.groupby('role'):
    s = g['sigma_T'].astype(float)
    print(role, 'n', len(g), 'nonzero', int((s.abs()>1e-6).sum()), 'max', float(s.max()))
PYEOF
