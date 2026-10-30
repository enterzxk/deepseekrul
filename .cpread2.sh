#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from pathlib import Path
import dataset
from dataset import build_piecewise_linear_rul, load_cmapss_train
df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
dataset.CP_LABELS = False
cap = np.asarray(build_piecewise_linear_rul(df, 125.0))
dataset.CP_LABELS = True
cp = np.asarray(build_piecewise_linear_rul(df, 125.0))
print("nan:", int(np.isnan(cp).sum()), "| range:", float(cp.min()), float(cp.max()))
print("frac rows changed:", round(float((np.abs(cap - cp) > 0.5).mean()), 3))
print("max abs diff:", float(np.abs(cap - cp).max()))
# no engine-specific scale: plateau equals min(T-t_cp,125), labels stay in cycles
g = df[df.unit == df.unit.iloc[0]].sort_values("cycle"); i = g.index
print("cap sample:", np.round(cap[i][::25], 0))
print("cp  sample:", np.round(cp[i][::25], 0))
EOF
