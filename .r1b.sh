
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import pandas as pd, glob, collections, numpy as np
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order03/*/*/*/matched_cutoff_predictions.csv"):
    tag = p.split("/")[2]; ds = p.split("/")[3]
    df = pd.read_csv(p)
    e = df["signed_error"].to_numpy()
    rows[(tag, ds)].append(dict(rmse=float(np.sqrt((e**2).mean())), late=float((e>0).mean()),
                                severe=int((e>10).sum()), nasa=float(np.sum(np.where(e<0, np.expm1(-e/13), np.expm1(e/10))))))
print(f"{'cell':14s} {'ds':6s} n  rmse   late   severe  nasa")
for k in sorted(rows):
    v = rows[k]; n = len(v)
    print(f"{k[0]:14s} {k[1]:6s} {n}  {np.mean([x['rmse'] for x in v]):6.3f} {np.mean([x['late'] for x in v]):5.3f}  {np.mean([x['severe'] for x in v]):6.1f}  {np.mean([x['nasa'] for x in v]):7.1f}")
EOF
echo "=== pending ==="
for ds in FD001 FD002; do for t in M0_raw_tf M1_global M2_pop M3_cond M4_racdr M4b_popgate M4c_noid; do
  for s in 2027 2028 2029; do
    ls experiments/order03/$t/$ds/*_s$s/metrics.json >/dev/null 2>&1 || echo "missing $t $ds $s"
  done
done; done | head -30
