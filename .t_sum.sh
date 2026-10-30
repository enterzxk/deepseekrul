
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob, collections
rows = collections.defaultdict(list)
for p in glob.glob("experiments/tournament/*/*/metrics.json"):
    d = json.load(open(p))
    rows[(d["subset"], d["variant"])].append(d["best_val_rmse"])
for (ds, v), vals in sorted(rows.items()):
    m = sum(vals)/len(vals)
    print(f"{ds} {v:5s} n={len(vals)} mean={m:7.3f}  " + " ".join(f"{x:.3f}" for x in sorted(vals)))
print("--- running ---")
EOF
pgrep -c -f run_candidate
