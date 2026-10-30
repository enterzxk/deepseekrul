#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python .patch3.py
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from pathlib import Path
import dataset
from dataset import build_piecewise_linear_rul, load_cmapss_train
df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
dataset.CP_LABELS = True
cp = np.asarray(build_piecewise_linear_rul(df, 125.0))
print("nan count:", int(np.isnan(cp).sum()), "min:", float(cp.min()), "max:", float(cp.max()))
dataset.CP_LABELS = False
cap = np.asarray(build_piecewise_linear_rul(df, 125.0))
print("frac rows changed:", float((np.abs(cap-cp)>0.5).mean()))
import ast; ast.parse(open("src/dataset.py", encoding="utf-8").read()); print("syntax ok")
EOF
