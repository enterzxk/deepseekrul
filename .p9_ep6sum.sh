
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import json, glob, numpy as np, csv
rows = {}
for path in glob.glob("experiments/p9/final_ep6/FD002/*/seed_*/p9_*/metrics.json") + glob.glob("experiments/p9/final_ep6/FD002/*/seed_*/*/metrics.json"):
    d = json.load(open(path)); tm = d["test_metrics"]
    rows.setdefault(d["variant"], []).append((d["seed"], tm["rmse"], tm["nasa_score"]))
for v, vals in sorted(rows.items()):
    a = np.array([x[1] for x in vals])
    print(f"{v:20s} n={len(a)} rmse={a.mean():6.3f}+/-{a.std():.3f} nasa={np.mean([x[2] for x in vals]):7.1f}")
# ensemble
for v in ("p9_full", "transformer"):
    paths = sorted(glob.glob(f"experiments/p9/final_ep6/FD002/{v}_s*/seed_*/{v}/test_predictions.csv"))
    if len(paths) < 2: continue
    ps, ys = [], None
    for p in paths:
        r = list(csv.DictReader(open(p)))
        ps.append(np.array([float(x["y_pred"]) for x in r]))
        ys = ys if ys is not None else np.array([float(x["y_true"]) for x in r])
    e = np.mean(ps, axis=0) - ys
    print(f"ENSEMBLE {v:16s} rmse={np.sqrt(np.mean(e**2)):.3f} nasa={np.sum(np.where(e<0, np.expm1(-e/13), np.expm1(e/10))):.1f}")
EOF
