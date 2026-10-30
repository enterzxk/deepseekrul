
cd /home/yndx/zxk/bearing-rul-pi
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, numpy as np, pandas as pd
def per(dfs, col):
    d = pd.concat(dfs)
    return d.groupby(col).signed_error.apply(lambda s: float(np.sqrt((s**2).mean())))
def load(tag, kind):
    name = "pseudo_test_predictions.csv" if kind=="B" else "protocol_c_predictions.csv"
    return [pd.read_csv(f) for f in glob.glob(f"experiments/order03/{tag}/FD002/*/{name}")]
print("=== Protocol C (absolute cycle) RMSE by cutoff, FD002 ===")
tab = {}
for tag in ("M0_raw_tf","M6_tfA","M1_global","M5_globalgate","M4_racdr","M3_cond"):
    dfs = load(tag,"C")
    if dfs: tab[tag] = per(dfs,"abs_cut")
cols = sorted(tab["M1_global"].index)
print(f"{'cell':14s} " + " ".join(f"{c:>7}" for c in cols))
for tag,v in tab.items():
    print(f"{tag:14s} " + " ".join(f"{v.get(c,float('nan')):7.3f}" for c in cols))
print()
print("=== Protocol B (life fraction) RMSE by cutoff, FD002 ===")
tab2 = {}
for tag in ("M0_raw_tf","M6_tfA","M1_global","M5_globalgate","M4_racdr","M3_cond"):
    dfs = load(tag,"B")
    if dfs: tab2[tag] = per(dfs,"cutoff")
cols2 = sorted(tab2["M1_global"].index)
print(f"{'cell':14s} " + " ".join(f"{c:>7}" for c in cols2))
for tag,v in tab2.items():
    print(f"{tag:14s} " + " ".join(f"{v.get(c,float('nan')):7.3f}" for c in cols2))
EOF
