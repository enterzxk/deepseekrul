
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob
for tag in ("M1_global","M5_globalgate","M6_tfA"):
    for p in sorted(glob.glob(f"experiments/order03/{tag}/FD002/*/metrics.json")):
        d = json.load(open(p))
        print(f"{tag:14s} seed={d['model_seed']} matched={d.get('matched_cutoff_rmse'):.3f} val={d['best_val_rmse']:.3f}")
EOF
