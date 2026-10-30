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
print("label correlation (cap vs cp):", round(float(np.corrcoef(cap, cp)[0,1]), 4))
print("mean |diff|:", round(float(np.abs(cap-cp).mean()), 2), " max |diff|:", float(np.abs(cap-cp).max()))
print("frac rows |diff|>5:", round(float((np.abs(cap-cp)>5).mean()), 3))
# how much of the label variance is shared
print("R^2 of linear fit cp~cap:", round(float(1 - np.var(cp-cap)/np.var(cp)), 4))
EOF
