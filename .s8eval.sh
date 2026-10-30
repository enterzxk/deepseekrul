
cd /home/yndx/zxk/bearing-rul-pi
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_resampled_metric.py --root experiments/split8 --out results/dsh/split8 --ref M6_tfA 2>&1 | tail -12
echo "=== safety ==="
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, numpy as np, pandas as pd
for t in ("M0_raw_tf","M6_tfA","M1_global"):
    fs = glob.glob(f"experiments/split8/{t}/FD002/*/matched_cutoff_predictions.csv")
    e = np.concatenate([pd.read_csv(f).signed_error.to_numpy() for f in fs])
    print(f"{t:12s} rmse={np.sqrt((e**2).mean()):6.3f} late={(e>0).mean():.3f} severe={int((e>10).sum())}")
EOF
