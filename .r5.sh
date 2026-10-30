
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
echo "trainers=$(pgrep -c -f '[r]un_candidate') controllers=$(pgrep -fc '[r]4q.sh')"
python - <<'EOF'
import json, glob, collections, numpy as np, pandas as pd
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order03/*/*/*/metrics.json"):
    d = json.load(open(p)); rows[(p.split("/")[2], d["subset"])].append(d)
print(f"{'cell':16s} {'ds':6s} n  matched")
for k in sorted(rows):
    v = rows[k]
    print(f"{k[0]:16s} {k[1]:6s} {len(v)}  {np.nanmean([x.get('matched_cutoff_rmse', np.nan) for x in v]):7.3f}")
print("--- M1 vs M4 per-seed (FD002) ---")
for tag in ("M1_global","M4_racdr","M3_cond"):
    for p in sorted(glob.glob(f"experiments/order03/{tag}/FD002/*/metrics.json")):
        d = json.load(open(p))
        print(f"{tag:10s} seed={d['model_seed']} matched={d.get('matched_cutoff_rmse'):.3f} val={d['best_val_rmse']:.3f}")
EOF
ls experiments/order03/M4_ckpt/FD002/*/best_model.pt 2>/dev/null | wc -l
