#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import run_candidate as rc
df = rc.load_cmapss_train(rc.ROOT / "data" / "CMAPSS", "FD002")
g = df[df.unit == df.unit.iloc[0]].sort_values("cycle")
rc.CP_MODE = False
a = rc.build_piecewise_linear_rul(g, 125.0)
rc.CP_MODE = True
b = rc.build_piecewise_linear_rul(g, 125.0)
print("labels off (head):", np.round(a[:6], 1))
print("labels on  (head):", np.round(b[:6], 1))
print("max abs diff:", float(np.abs(a - b).max()))
print("identical:", bool(np.allclose(a, b)))
EOF
