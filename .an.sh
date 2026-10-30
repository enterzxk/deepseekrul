
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import glob, json, numpy as np, pandas as pd
def cells(tag, ds="FD002"):
    return sorted(glob.glob(f"experiments/order03/{tag}/{ds}/*/metrics.json"))
def proto(tag, ds="FD002"):
    out = {"B_pooled":[], "B_macro":[], "C_pooled":[], "C_macro":[], "matched":[]}
    for p in cells(tag, ds):
        d = json.load(open(p))
        b = d.get("protocol_b_rmse_by_cutoff") or d.get("pseudo_test_rmse_by_cutoff") or {}
        c = d.get("protocol_c_rmse_by_cycles") or {}
        if b: out["B_pooled"].append(np.mean(list(b.values()))); out["B_macro"].append(np.mean(list(b.values())))
        if c: out["C_pooled"].append(np.mean(list(c.values()))); out["C_macro"].append(np.mean(list(c.values())))
        if d.get("matched_cutoff_rmse"): out["matched"].append(d["matched_cutoff_rmse"])
    return out
print(f"{'cell':10s} {'B_pooled':>9s} {'B_macro':>9s} {'C_pooled':>9s} {'matched':>9s}")
for tag in ("M0_raw_tf","M6_tfA","M1_global","M5_globalgate","M4_racdr","M2_pop","M3_cond"):
    o = proto(tag)
    if not o["matched"]: continue
    f = lambda v: np.mean(v) if v else float("nan")
    print(f"{tag:10s} {f(o['B_pooled']):9.3f} {f(o['B_macro']):9.3f} {f(o['C_pooled']):9.3f} {f(o['matched']):9.3f}")
print()
print("=== band errors (matched, FD002, pooled over seeds) ===")
for tag in ("M0_raw_tf","M6_tfA","M1_global","M4_racdr"):
    fs = glob.glob(f"experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv")
    if not fs: continue
    df = pd.concat([pd.read_csv(f) for f in fs])
    bands = [(0,30),(30,60),(60,90),(90,126)]
    row = []
    for lo,hi in bands:
        m = (df.target>=lo)&(df.target<hi)
        row.append(f"[{lo},{hi}):{np.sqrt((df.signed_error[m]**2).mean()):.2f}(n={m.sum()})" if m.sum() else f"[{lo},{hi}):-")
    print(f"{tag:10s} " + " ".join(row))
EOF
