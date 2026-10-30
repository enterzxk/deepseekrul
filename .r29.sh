
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py') done=$(ls experiments/shuf/*/FD002/*/metrics.json 2>/dev/null | wc -l)"
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, json, numpy as np
for p in sorted(glob.glob("experiments/shuf/A/FD002/*/metrics.json")):
    d=json.load(open(p)); print(f"shuf seed={d['model_seed']} val={d['best_val_rmse']:.2f} matched={d.get('matched_cutoff_rmse',float('nan')):.2f}")
for p in sorted(glob.glob("experiments/order03/M1_global/FD002/*/metrics.json")):
    d=json.load(open(p)); print(f"aligned seed={d['model_seed']} val={d['best_val_rmse']:.2f} matched={d.get('matched_cutoff_rmse',float('nan')):.2f}")
EOF
