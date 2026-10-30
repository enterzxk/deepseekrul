
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f '[r]un_candidate') cells=$(ls experiments/order03/*/*/*/metrics.json | wc -l)"
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob, collections, numpy as np
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order03/*/*/*/metrics.json"):
    d = json.load(open(p)); rows[(p.split("/")[2], d["subset"])].append(d)
for k in sorted(rows):
    v = rows[k]; m=[x.get("matched_cutoff_rmse") for x in v if x.get("matched_cutoff_rmse")]
    if k[0] in ("M1_global","M6_tfA","M7_rawback","M0_raw_tf"):
        print(f"{k[0]:12s} {k[1]} n={len(v)} matched={np.mean(m):7.3f} seeds={[x['model_seed'] for x in v]}")
EOF
