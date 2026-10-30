
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob, collections, os
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order01/*/*/metrics.json"):
    d = json.load(open(p))
    rows[(d["subset"], d["variant"])].append((d["model_seed"], d["best_val_rmse"],
        d.get("pseudo_test_rmse_mean"), d["best_epoch"], d["restored_best_before_pseudo_test"]))
for (ds, v), vals in sorted(rows.items()):
    vm = sum(x[1] for x in vals)/len(vals); pm = sum(x[2] for x in vals)/len(vals)
    print(f"{ds} {v:5s} n={len(vals)} val={vm:7.3f} pseudo={pm:7.3f} seeds={[x[0] for x in vals]} restored={all(x[4] for x in vals)}")
EOF
echo "--- missing ---"
for ds in FD001 FD002; do for v in tf tfA A Aid Arec; do for s in 2027 2028 2029; do
  f=experiments/order01/$ds/${v}_s${s}/metrics.json
  [ -f "$f" ] || echo "$ds $v $s"
done; done; done
