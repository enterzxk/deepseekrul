
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py')"
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, numpy as np, pandas as pd, json
def ms(tag, sub="FD002"):
    fs = glob.glob(f"experiments/order03/{tag}/{sub}/*/matched_cutoff_predictions.csv")
    if not fs: return None
    e = np.concatenate([pd.read_csv(f).signed_error.to_numpy() for f in fs])
    return len(fs), np.sqrt((e**2).mean()), (e>0).mean(), int((e>10).sum())
for t in ("M0_raw_tf","M7_rawback","M6_tfA","M1_global"):
    r = ms(t)
    print(f"{t:12s} n={r[0]} rmse={r[1]:6.3f} late={r[2]:.3f} severe={r[3]}")
EOF
