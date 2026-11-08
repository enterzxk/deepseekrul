#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY - <<'PYEOF'
import pandas as pd, numpy as np
df = pd.read_csv('experiments/ida/prior/FD001_prior_cutoffs.csv')
print('columns:', list(df.columns))
print('rows', len(df))
for c in df.columns:
    if 'sig' in c.lower() or 'mu' in c.lower():
        s = pd.to_numeric(df[c], errors='coerce')
        print(c, 'min', round(float(s.min()),4), 'median', round(float(s.median()),4), 'max', round(float(s.max()),4), 'zeros', int((s==0).sum()))
print(df.head(3).to_string()[:400])
PYEOF
