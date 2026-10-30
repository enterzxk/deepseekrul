#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo "trainers=$(pgrep -c -f '[r]un_candidate') done=$(ls experiments/cplab2/FD002/*/metrics.json 2>/dev/null | wc -l)"
cat /home/yndx/p8/cp2.state
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, json
for p in sorted(glob.glob("experiments/cplab2/FD002/*/metrics.json")):
    d=json.load(open(p)); print(f"cplab2 seed={d['model_seed']} val={d['best_val_rmse']:.2f} matched={d.get('matched_cutoff_rmse',float('nan')):.2f}")
EOF
