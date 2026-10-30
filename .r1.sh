
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
echo "=== procs ==="; pgrep -c -f "[r]un_candidate"
echo "=== cells ==="; ls experiments/order03/*/*/*/metrics.json 2>/dev/null | wc -l
echo "=== table ==="
python - <<'EOF'
import json, glob, collections
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order03/*/*/*/metrics.json"):
    d = json.load(open(p)); tag = p.split("/")[2]
    rows[(tag, d["subset"], d["variant"])].append(d)
for k in sorted(rows):
    v = rows[k]; n = len(v)
    val = sum(x["best_val_rmse"] for x in v)/n
    mat = sum(x.get("matched_cutoff_rmse", float("nan")) for x in v)/n
    pc = [x.get("protocol_c_rmse_by_cycles", {}) for x in v]
    pcm = sum(sum(p.values())/max(1,len(p)) for p in pc)/n
    print(f"{k[0]:14s} {k[1]} {k[2]:11s} n={n} val={val:7.3f} matched={mat:7.3f} protoC={pcm:7.3f}")
EOF
echo "=== code facts ==="
grep -n '"RACDR"' -A4 scripts/run_candidate.py | head -8
grep -n "use_sensor_id" src/candidates/model.py | head -6
grep -n "e4" scripts/run_candidate.py | head -4
