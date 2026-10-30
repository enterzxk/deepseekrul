
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob
for p in sorted(glob.glob("experiments/p10/gate/*/seed_2027/*/metrics.json")):
    d = json.load(open(p)); t = d["test_metrics"]
    print(f'{p.split("/")[3]:22s} val={d["best_validation_rmse"]:6.3f} test={t["rmse"]:6.3f} nasa={t["nasa_score"]:7.1f} mono={t["mono_violation_rate"]:.3f} params={d["num_parameters"]:,}')
EOF
