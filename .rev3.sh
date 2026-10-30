#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import sys, numpy as np
sys.path.insert(0, "src")
from pathlib import Path
from dataset import build_piecewise_linear_rul, load_cmapss_train
df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
g = df[df.unit == df.unit.iloc[0]].sort_values("cycle")
a = np.asarray(build_piecewise_linear_rul(g, 125.0))
print("head:", np.round(a[:5], 1), "tail:", np.round(a[-5:], 1), "max:", a.max())
print("restored to standard cap labels:", bool(a.max() == 125.0 and a[-1] == 0.0))
EOF
