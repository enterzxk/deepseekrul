
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob
for p in sorted(glob.glob("experiments/p10/sweep/*/seed_2027/p10_full/metrics.json")):
    d = json.load(open(p)); t = d["test_metrics"]
    print(f'{p.split("/")[3]:8s} val={d["best_validation_rmse"]:6.3f} test={t["rmse"]:6.3f} nasa={t["nasa_score"]:7.1f} ep={d["best_epoch"]:2d}')
EOF
