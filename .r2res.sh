cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f '[r]un_candidate') done=$(ls experiments/M3z/FD002/*/metrics.json 2>/dev/null | wc -l)"
cat /home/yndx/p8/r2.state 2>/dev/null
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, json
for p in sorted(glob.glob("experiments/M3z/FD002/*/metrics.json")):
    d=json.load(open(p)); print(f"M3z seed={d['model_seed']} val={d['best_val_rmse']:.3f} matched={d.get('matched_cutoff_rmse',float('nan')):.3f}")
for p in sorted(glob.glob("experiments/order03/M3_cond/FD002/*/metrics.json")):
    d=json.load(open(p)); print(f"M3  seed={d['model_seed']} val={d['best_val_rmse']:.3f} matched={d.get('matched_cutoff_rmse',float('nan')):.3f}")
EOF