
cd /home/yndx/zxk/bearing-rul-pi
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, numpy as np, pandas as pd
def ms(tag, sub):
    fs = glob.glob(f"experiments/order03/{tag}/{sub}/*/matched_cutoff_predictions.csv")
    if not fs: return None
    e = np.concatenate([pd.read_csv(f).signed_error.to_numpy() for f in fs])
    return len(fs), float(np.sqrt((e**2).mean())), float((e>0).mean()), int((e>10).sum()), float(np.abs(e).mean())
print(f"{'cell/subset':22s} n   rmse   late  severe   mae")
for sub in ("FD001","FD002"):
    for t in ("M0_raw_tf","M7_rawback","M6_tfA","M1_global"):
        r = ms(t, sub)
        if r: print(f"{t+'/'+sub:22s} {r[0]}  {r[1]:6.3f} {r[2]:.3f} {r[3]:6d} {r[4]:6.3f}")
EOF
