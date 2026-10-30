
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob
for path in sorted(glob.glob("experiments/p9/gate/*/seed_2027/*/metrics.json")):
    d = json.load(open(path)); tm = d["test_metrics"]
    print(f'{d["subset"]} {d["variant"]:12s} val={d["best_validation_rmse"]:6.3f} test={tm["rmse"]:6.3f} nasa={tm["nasa_score"]:7.1f} mae={tm["mae"]:5.2f} mono={tm["mono_violation_rate"]:.3f} ep={d["best_epoch"]:2d} params={d["num_parameters"]:,}')
EOF
