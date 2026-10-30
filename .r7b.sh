
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob, collections, numpy as np
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order03/*/*/*/metrics.json"):
    d = json.load(open(p)); rows[(p.split("/")[2], d["subset"])].append(d)
print("=== FD002 2x2 decomposition (matched-cutoff) ===")
for tag in ("M0_raw_tf","M6_tfA","M7_rawback","M1_global","M4_racdr","M5_globalgate","M2_pop","M3_cond"):
    v = rows.get((tag,"FD002"), [])
    if not v: continue
    m = [x.get("matched_cutoff_rmse") for x in v if x.get("matched_cutoff_rmse")]
    print(f"{tag:14s} n={len(v)} matched_mean={np.mean(m):7.3f} std={np.std(m):5.3f} per_seed={[round(x,2) for x in m]}")
EOF
