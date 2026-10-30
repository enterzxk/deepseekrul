
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob, collections, numpy as np
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order03/*/*/*/metrics.json"):
    d = json.load(open(p)); rows[(p.split("/")[2], d["subset"])].append(d)
for k in sorted(rows):
    v = rows[k]; n = len(v)
    print(f"{k[0]:14s} {k[1]} n={n} val={np.mean([x['best_val_rmse'] for x in v]):7.3f} matched={np.nanmean([x.get('matched_cutoff_rmse', np.nan) for x in v]):7.3f}")
EOF
echo "cells=$(ls experiments/order03/*/*/*/metrics.json | wc -l) trainers=$(pgrep -c -f '[r]un_candidate')"
