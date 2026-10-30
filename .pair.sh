
cd /home/yndx/zxk/bearing-rul-pi
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, numpy as np, pandas as pd
def eng(tag, sub="FD002"):
    fs = glob.glob(f"experiments/order03/{tag}/{sub}/*/protocol_c_predictions.csv")
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    return {u: g.set_index("abs_cut").signed_error.to_dict() for u,g in d.groupby("unit_id")}
pairs = [("M4b_popgate","H2_fixed1"), ("M4b_popgate","M1_global"), ("M4b_popgate","M4_racdr"), ("M4b_popgate","M6_tfA")]
for a,b in pairs:
    ea, eb = eng(a), eng(b)
    if not ea or not eb: 
        print(f"{a} vs {b}: missing"); continue
    units = sorted(set(ea) & set(eb))
    rng = np.random.default_rng(20261022); B=2000
    da=[]
    for _ in range(B):
        ea_v=[];eb_v=[]
        for u in units:
            c = rng.choice(list(ea[u].keys()))
            ea_v.append(ea[u][c]); eb_v.append(eb[u].get(c, np.nan))
        ea_v=np.array(ea_v); eb_v=np.array(eb_v); m=~np.isnan(eb_v)
        da.append(np.sqrt((ea_v[m]**2).mean())-np.sqrt((eb_v[m]**2).mean()))
    da=np.array(da); lo,hi=np.percentile(da,[2.5,97.5])
    print(f"{a} vs {b}: P(better)={float((da<0).mean()):.3f} median={np.median(da):+.3f} CI=[{lo:+.3f},{hi:+.3f}] ctx={len(units)}")
EOF
