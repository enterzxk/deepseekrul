#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo "trainers=$(pgrep -c -f '[r]un_candidate')"
cat /home/yndx/p8/cp.state 2>/dev/null
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, json
for p in sorted(glob.glob("experiments/cplabels/FD002/*/metrics.json")):
    d=json.load(open(p))
    print(f"cp seed={d['model_seed']} val={d['best_val_rmse']:.2f} epochs={d.get('epochs_ran')}")
EOF
