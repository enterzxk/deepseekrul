
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import glob, numpy as np, pandas as pd
def stat(tag):
    fs = glob.glob(f"experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv")
    if not fs: return None
    es = [pd.read_csv(f)["signed_error"].to_numpy() for f in fs]
    e = np.concatenate(es)
    return dict(n=len(fs), rmse=float(np.sqrt((e**2).mean())), late=float((e>0).mean()),
                severe=int((e>10).sum()), nasa=float(np.sum(np.where(e<0,np.expm1(-e/13),np.expm1(e/10)))))
print(f"{'cell':14s} seeds  rmse   late  severe  nasa")
for tag in ("M0_raw_tf","M6_tfA","M1_global","M5_globalgate","M4_racdr"):
    r = stat(tag)
    if r: print(f"{tag:14s} {r['n']}    {r['rmse']:6.3f} {r['late']:.3f} {r['severe']:6d} {r['nasa']:7.1f}")
EOF
