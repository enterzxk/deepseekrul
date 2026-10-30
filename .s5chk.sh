
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py')"
cat /home/yndx/p8/shuf5.state 2>/dev/null
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, json
for p in sorted(glob.glob("experiments/shuf_fd001/FD001/A_s*/metrics.json")):
    d=json.load(open(p)); print(f"shuf_FD001 seed={d['model_seed']} val={d['best_val_rmse']:.2f} matched={d.get('matched_cutoff_rmse',float('nan')):.2f}")
EOF
