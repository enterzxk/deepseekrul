
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob
for p in sorted(glob.glob("experiments/order02b/*/*/metrics.json")):
    d = json.load(open(p))
    print(d["subset"], d["variant"], d["model_seed"], "val=%.3f" % d["best_val_rmse"],
          "matched=%.3f" % d.get("matched_cutoff_rmse", -1),
          "ep=%d" % d["epochs_ran"])
EOF
pgrep -c -f run_candidate
