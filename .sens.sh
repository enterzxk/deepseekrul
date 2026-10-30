
cd /home/yndx/zxk/bearing-rul-pi
echo "=== checkpoints ==="
ls experiments/order03/M1_global/FD002/*/best_model.pt 2>/dev/null | wc -l
ls experiments/order03/M4_ckpt/FD002/*/best_model.pt 2>/dev/null | wc -l
ls experiments/order03/M6_tfA/FD002/*/best_model.pt 2>/dev/null | wc -l
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import glob, numpy as np, pandas as pd
def load_c(tag):
    fs = glob.glob(f"experiments/order03/{tag}/FD002/*/protocol_c_predictions.csv")
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else None
tags = ["M0_raw_tf","M6_tfA","M1_global","M5_globalgate","M4_racdr"]
data = {t: load_c(t) for t in tags}
data = {t:d for t,d in data.items() if d is not None}
rng = np.random.default_rng(7)
B = 2000
res = {t: [] for t in data}
# per engine, sample one of its evaluated cutoffs uniformly (deployment-like)
eng_cut = {}
for t,d in data.items():
    for u,g in d.groupby("unit_id"):
        eng_cut.setdefault((t,u), g.set_index("abs_cut").signed_error.to_dict())
units = sorted({u for (_,u) in eng_cut})
for b in range(B):
    for t,d in data.items():
        errs = []
        for u in units:
            m = eng_cut.get((t,u))
            if not m: continue
            c = rng.choice(list(m.keys()))
            errs.append(m[c])
        res[t].append(float(np.sqrt(np.mean(np.square(errs)))))
print(f"{'cell':14s} mean   p05    p95")
for t in data:
    a = np.array(res[t]); print(f"{t:14s} {a.mean():6.3f} {np.percentile(a,5):6.3f} {np.percentile(a,95):6.3f}")
for a,b in (("M1_global","M4_racdr"),("M1_global","M6_tfA"),("M4_racdr","M6_tfA")):
    if a in res and b in res:
        d = np.array(res[a]) - np.array(res[b])
        print(f"P({a} better than {b}) = {float((d<0).mean()):.3f}  median diff={np.median(d):+.3f}")
EOF
