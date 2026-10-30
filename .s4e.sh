
cd /home/yndx/zxk/bearing-rul-pi
find experiments/shuf -name metrics.json 2>/dev/null | head
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, json, numpy as np, pandas as pd
for p in sorted(glob.glob("experiments/shuf/FD002/A_s*/metrics.json")):
    d=json.load(open(p)); print(f"shuffled seed={d['model_seed']} val={d['best_val_rmse']:.2f} matched={d.get('matched_cutoff_rmse',float('nan')):.2f}")
EOF
