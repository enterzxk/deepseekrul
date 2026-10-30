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
dataset.CP_LABELS = False; cap = np.asarray(build_piecewise_linear_rul(df, 125.0))
dataset.CP_LABELS = True;  cp  = np.asarray(build_piecewise_linear_rul(df, 125.0))
d = np.abs(cap - cp)
print("nan:", int(np.isnan(cp).sum()), "| range:", float(cp.min()), float(cp.max()))
print("frac rows changed >0.5:", round(float((d > 0.5).mean()), 4))
print("frac rows changed >5  :", round(float((d > 5).mean()), 4))
print("max abs diff:", float(d.max()))
print("label corr:", round(float(np.corrcoef(cap, cp)[0,1]), 4))
EOF
