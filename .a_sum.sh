
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob, collections
rows = collections.defaultdict(list)
for p in glob.glob("experiments/trackA/*/*/metrics.json"):
    d = json.load(open(p))
    rows[(d["subset"], d["variant"])].append((d["best_val_rmse"], d.get("pseudo_test_rmse")))
for (ds, v), vals in sorted(rows.items()):
    vmean = sum(x[0] for x in vals)/len(vals)
    pmean = sum(x[1] for x in vals)/len(vals) if vals[0][1] is not None else float("nan")
    print(f"{ds} {v:5s} n={len(vals)} val={vmean:7.3f} pseudo={pmean:7.3f}")
EOF
pgrep -c -f run_candidate
