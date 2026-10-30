"""Pool validation engines across splits: each engine is evaluated by the model
that did NOT train on it, so M1 vs M6 remains a within-engine paired comparison.

Coverage: an engine appears in validation for whichever split(s) excluded it.
If an engine appears in several splits we average its per-split RMSE (disclosed).
This raises the paired sample from 52 to the union of validation engines, at the
cost of mixing model instances trained on different data.
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOTS = {"split7": "experiments/order03", "split8": "experiments/split8", "split9": "experiments/split9"}
OUT = Path("results/dsh/review_r1")

def per_engine(root, tag):
    fs = glob.glob(f"{root}/{tag}/FD002/*/matched_cutoff_predictions.csv")
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f).assign(model_seed=int(Path(f).parts[-2].split("_s")[-1])) for f in fs])
    d["sq"] = d.signed_error.astype(float) ** 2
    per = d.groupby(["unit_id", "model_seed"]).sq.mean().reset_index()
    return per.groupby("unit_id").sq.mean().pow(0.5)

frames = []
for name, root in ROOTS.items():
    a, b = per_engine(root, "M1_global"), per_engine(root, "M6_tfA")
    if a is None or b is None:
        continue
    m = a.to_frame("m1").join(b.to_frame("m6"), how="inner")
    m["split"] = name
    frames.append(m)
allm = pd.concat(frames)
print("paired observations across splits:", len(allm), "| unique engines:", allm.index.nunique())

pooled = allm.groupby(level=0)[["m1", "m6"]].mean()   # average if engine in >1 split
x, y = pooled.m1.to_numpy(), pooled.m6.to_numpy()
rng = np.random.default_rng(20261022)
d = []
for _ in range(5000):
    t = rng.integers(0, len(x), len(x))
    d.append(np.sqrt((x[t] ** 2).mean()) - np.sqrt((y[t] ** 2).mean()))
d = np.asarray(d)
out = {"unique_engines": int(len(pooled)),
       "m1_rmse": float(np.sqrt((x ** 2).mean())), "m6_rmse": float(np.sqrt((y ** 2).mean())),
       "median_diff": float(np.median(d)),
       "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
       "prob_m1_better": float((d < 0).mean()),
       "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
       "caveat": "engines evaluated by models trained on different data; not independent replicates"}
print(json.dumps(out, indent=2))
json.dump(out, open(OUT / "pooled_engines.json", "w"), indent=2)
pooled.to_csv(OUT / "pooled_engine_rmse.csv")
