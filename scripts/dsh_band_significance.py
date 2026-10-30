"""Is the early-life advantage of M1 over M6 significant within the band?

The aggregate ranking mixes RUL bands. This tests the difference *within* each band
with an engine-clustered bootstrap, pooling the three splits but resampling engines
(with the overlap caveat stated in the output).
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOTS = {"split7": "experiments/order03", "split8": "experiments/split8", "split9": "experiments/split9"}
OUT = Path("results/dsh/review_r1")

def load(root, tag):
    fs = glob.glob(f"{root}/{tag}/FD002/*/matched_cutoff_predictions.csv")
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f).assign(model_seed=int(Path(f).parts[-2].split("_s")[-1])) for f in fs])
    d["sq"] = d.signed_error.astype(float) ** 2
    per = d.groupby(["unit_id", "model_seed"]).sq.mean().reset_index()
    per["rmse"] = np.sqrt(per.sq)
    rul = d.groupby("unit_id").y_true.mean()
    return per.groupby("unit_id").rmse.mean().to_frame("rmse").join(rul.rename("rul"), how="left")

rows = []
for name, root in ROOTS.items():
    a, b = load(root, "M1_global"), load(root, "M6_tfA")
    if a is None or b is None:
        continue
    m = a.join(b, lsuffix="_m1", rsuffix="_m6", how="inner")
    m["split"] = name
    rows.append(m.reset_index())
df = pd.concat(rows, ignore_index=True)

def band_boot(sub, lo, hi, reps=5000, seed=20261022):
    s = sub[(sub.rul_m1 >= lo) & (sub.rul_m1 < hi)]
    if len(s) < 8:
        return None
    x = s.rmse_m1.to_numpy(); y = s.rmse_m6.to_numpy()
    rng = np.random.default_rng(seed); d = []
    for _ in range(reps):
        t = rng.integers(0, len(x), len(x))
        d.append(np.sqrt((x[t] ** 2).mean()) - np.sqrt((y[t] ** 2).mean()))
    d = np.asarray(d)
    return {"band": f"{lo}-{hi}", "n": int(len(s)),
            "m1": float(np.sqrt((x ** 2).mean())), "m6": float(np.sqrt((y ** 2).mean())),
            "median_diff": float(np.median(d)),
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "prob_m1_better": float((d < 0).mean()),
            "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}

res = [band_boot(df, lo, hi) for lo, hi in [(0, 40), (40, 80), (80, 126)]]
res = [r for r in res if r]
for r in res: print(json.dumps(r))
json.dump({"bands": res, "note": "three splits share engines; observations not independent"},
          open(OUT / "band_significance.json", "w"), indent=2)
