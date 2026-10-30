
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
echo "trainers=$(pgrep -c -f '[r]un_candidate')"
python - <<'EOF'
import json, glob, numpy as np
for tag in ("M6_tfA","M1_global"):
    vals = []
    for p in sorted(glob.glob(f"experiments/order03/{tag}/FD002/*/metrics.json")):
        d = json.load(open(p)); vals.append(d.get("matched_cutoff_rmse"))
        print(f"{tag:12s} seed={d['model_seed']} matched={d.get('matched_cutoff_rmse'):.3f} val={d['best_val_rmse']:.3f}")
    if vals: print(f"{tag} mean matched = {np.mean([v for v in vals if v]):.3f} (n={len(vals)})")
EOF
