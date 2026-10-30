cd /home/yndx/zxk/bearing-rul-pi
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, json, numpy as np
def get(tag):
    out={}
    for p in sorted(glob.glob(f"experiments/{tag}/FD002/*/metrics.json")):
        d=json.load(open(p)); out[d['model_seed']]=(d['best_val_rmse'], d.get('matched_cutoff_rmse'))
    return out
m3z, m3, m1 = get("M3z"), get("order03/M3_cond"), get("order03/M1_global")
for name,d in (("M3z(norm+cond)",m3z),("M3(cond,no norm)",m3),("M1(global)",m1)):
    if not d: continue
    vals=[v[1] for v in d.values() if v[1]]
    print(f"{name:20s} n={len(d)} matched={np.mean(vals):7.3f}  per_seed={[round(v,2) for v in vals]}")
print()
a=np.mean([v[1] for v in m3z.values()])
print(f"decision inputs: M3z={a:.2f}  baseline M3=16.06  M1=12.93")
print(f"prediction1 (M3z < 16.41): {a < 16.41}")
print(f"prediction2 (|M3z - 12.63| <= 1.0): {abs(a-12.63) <= 1.0}")
print(f"prediction3 (M3z > 13.6 -> scale rejected): {a > 13.6}")
EOF