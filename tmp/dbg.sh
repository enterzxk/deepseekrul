#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY - <<'PYEOF'
import sys, json
sys.path.insert(0, '.'); sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_ida as R
import numpy as np
folds, df = R.outer_folds('FD001')
tr_u, va_u = folds[0]
(xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = R.prep_fold(df, 'FD001', 0, tr_u, va_u, window=50)
print('train', len(xtr), len(ytr), mtr.shape)
print('val', len(xva), len(yva), mva.shape)
PYEOF
