
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
echo "trainers=$(pgrep -c -f '[r]un_candidate') cells=$(ls experiments/order03/*/*/*/metrics.json | wc -l)"
echo "ckpts=$(ls experiments/order03/M4_ckpt/FD002/*/best_model.pt 2>/dev/null | wc -l)"
python - <<'EOF'
import json, glob, collections, numpy as np, pandas as pd
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order03/*/*/*/metrics.json"):
    d = json.load(open(p)); rows[(p.split("/")[2], d["subset"])].append(d)
print(f"{'cell':16s} {'ds':6s} n  matched   val")
for k in sorted(rows):
    v = rows[k]
    print(f"{k[0]:16s} {k[1]:6s} {len(v)}  {np.nanmean([x.get('matched_cutoff_rmse', np.nan) for x in v]):7.3f} {np.mean([x['best_val_rmse'] for x in v]):7.3f}")
# safety for H2 vs M4 on FD002
def safe(pat):
    fs = glob.glob(pat)
    if not fs: return None
    es = [pd.read_csv(f)["signed_error"].to_numpy() for f in fs]
    e = np.concatenate(es)
    return dict(rmse=float(np.sqrt((e**2).mean())), late=float((e>0).mean()), severe=int((e>10).sum()), n=len(fs))
print("--- H2 vs M4 (FD002 matched, pooled) ---")
for tag in ("M4_racdr", "M4_ckpt", "H2_fixed1", "H2_half", "H2_const"):
    r = safe(f"experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv")
    if r: print(f"{tag:12s} seeds={r['n']} rmse={r['rmse']:.3f} late={r['late']:.3f} severe={r['severe']}")
EOF
