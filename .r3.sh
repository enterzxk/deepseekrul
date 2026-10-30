
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f '[r]un_candidate')"
ls experiments/order03/M4_ckpt/FD002/*/best_model.pt 2>/dev/null | wc -l
ls experiments/order03/H2_*/FD002/*/metrics.json 2>/dev/null | wc -l
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob, collections, numpy as np
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order03/*/*/*/metrics.json"):
    d = json.load(open(p)); rows[(p.split("/")[2], d["subset"])].append(d)
for k in sorted(rows):
    v = rows[k]
    print(f"{k[0]:14s} {k[1]} n={len(v)} matched={np.nanmean([x.get('matched_cutoff_rmse', np.nan) for x in v]):7.3f}")
EOF
