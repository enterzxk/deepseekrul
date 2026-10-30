#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import sys, numpy as np
sys.path.insert(0, "src")
from pathlib import Path
import dataset
from dataset import build_piecewise_linear_rul, load_cmapss_train
df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
dataset.CP_LABELS = False
cap = build_piecewise_linear_rul(df, 125.0)
dataset.CP_LABELS = True
cp = build_piecewise_linear_rul(df, 125.0)
print("max abs diff:", float(np.abs(np.asarray(cap) - np.asarray(cp)).max()))
print("frac rows changed:", float((np.abs(np.asarray(cap)-np.asarray(cp))>0.5).mean()))
g = df[df.unit == df.unit.iloc[0]].sort_values("cycle")
i = g.index
print("example labels cp (every 20th):", np.round(np.asarray(cp)[i][::20], 0))
print("example labels cap(every 20th):", np.round(np.asarray(cap)[i][::20], 0))
EOF
