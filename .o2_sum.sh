
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob, collections
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order02/*/*/metrics.json"):
    d = json.load(open(p))
    rows[(d["subset"], d["variant"])].append((d["best_val_rmse"], d["pseudo_test_rmse_mean"],
        sum(d["protocol_c_rmse_by_cycles"].values())/len(d["protocol_c_rmse_by_cycles"]),
        d["num_parameters"] if "num_parameters" in d else d.get("params")))
for (ds, v), vals in sorted(rows.items()):
    n = len(vals)
    print(f"{ds} {v:5s} n={n} val={sum(x[0] for x in vals)/n:7.3f} pseudoB={sum(x[1] for x in vals)/n:7.3f} protoC={sum(x[2] for x in vals)/n:7.3f} params={vals[0][3]:,}")
EOF
grep -c DONE ~/p8/o2.out
