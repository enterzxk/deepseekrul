#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import sys, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import dataset
from dataset import build_piecewise_linear_rul, load_cmapss_train
from pathlib import Path
df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
g = df[df.unit == df.unit.iloc[0]].sort_values("cycle")
dataset.LABEL_MODE = "cap"
a = build_piecewise_linear_rul(g, 125.0)
dataset.LABEL_MODE = "cp"
b = build_piecewise_linear_rul(g, 125.0)
print("max abs diff:", float(np.abs(np.asarray(a) - np.asarray(b)).max()))
print("cap head:", np.round(np.asarray(a)[:5], 1), " cp head:", np.round(np.asarray(b)[:5], 1))
print("cap tail:", np.round(np.asarray(a)[-5:], 1), " cp tail:", np.round(np.asarray(b)[-5:], 1))
EOF
