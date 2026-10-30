
cd /home/yndx/zxk/bearing-rul-pi
setsid nohup bash /home/yndx/p8/o3d.sh > /home/yndx/p8/o3e.out 2>&1 < /dev/null &
sleep 15
echo "controllers=$(pgrep -fc '[o]3d.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob, collections, numpy as np, pandas as pd
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order03/*/*/*/metrics.json"):
    d = json.load(open(p)); rows[(p.split("/")[2], d["subset"])].append(d)
for k in sorted(rows):
    v = rows[k]; n=len(v)
    print(f"{k[0]:14s} {k[1]} n={n} val={np.mean([x['best_val_rmse'] for x in v]):7.3f} matched={np.mean([x.get('matched_cutoff_rmse', np.nan) for x in v]):7.3f}")
EOF
