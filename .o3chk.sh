
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob, collections
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order03/*/*/*/metrics.json"):
    d = json.load(open(p))
    tag = p.split("/")[2]
    rows[(tag, d["subset"], d["variant"])].append((d["best_val_rmse"], d.get("matched_cutoff_rmse"),
        sum(d.get("protocol_c_rmse_by_cycles", {}).values())/max(1,len(d.get("protocol_c_rmse_by_cycles", {})))))
for k, v in sorted(rows.items()):
    n = len(v)
    print(f"{k[0]:12s} {k[1]} {k[2]:6s} n={n} val={sum(x[0] for x in v)/n:7.3f} matched={sum(x[1] for x in v)/n:7.3f} protoC={sum(x[2] for x in v)/n:7.3f}")
EOF
echo "--- procs ---"; pgrep -c -f run_candidate; pgrep -fc o3.sh
