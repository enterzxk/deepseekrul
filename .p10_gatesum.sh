
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob
for p in sorted(glob.glob("experiments/p10/gate/*/seed_2027/*/metrics.json")):
    d = json.load(open(p))
    print(f'{p.split("/")[3]:22s} best_epoch={d["best_epoch"]:2d} ran={d["epochs_ran"]:2d} val={d["best_validation_rmse"]:6.3f} test={d["test_metrics"]["rmse"]:6.3f}')
EOF
