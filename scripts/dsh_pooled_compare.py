"""Does the feature set help? M0 (raw) vs M6 (same backbone + reference features),
pooled over the union of validation engines across the three splits.
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

def compare(a_tag, b_tag):
    frames = []
    for name, root in ROOTS.items():
        a, b = per_engine(root, a_tag), per_engine(root, b_tag)
        if a is None or b is None:
            continue
        m = a.to_frame("a").join(b.to_frame("b"), how="inner")
        m["split"] = name
        frames.append(m)
    if not frames:
        return None
    allm = pd.concat(frames)
    pooled = allm.groupby(level=0)[["a", "b"]].mean()
    x, y = pooled.a.to_numpy(), pooled.b.to_numpy()
    rng = np.random.default_rng(20261022)
    d = []
    for _ in range(5000):
        t = rng.integers(0, len(x), len(x))
        d.append(np.sqrt((x[t] ** 2).mean()) - np.sqrt((y[t] ** 2).mean()))
    d = np.array(d)
    return {"a": a_tag, "b": b_tag, "engines": int(len(pooled)),
            "a_rmse": float(np.sqrt((x ** 2).mean())), "b_rmse": float(np.sqrt((y ** 2).mean())),
            "median_diff": float(np.median(d)),
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "prob_a_better": float((d < 0).mean()),
            "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}

res = []
for a, b in [("M6_tfA", "M0_raw_tf"), ("M1_global", "M0_raw_tf"), ("M1_global", "M6_tfA")]:
    r = compare(a, b)
    if r:
        res.append(r); print(json.dumps(r))
json.dump(res, open(OUT / "pooled_comparisons.json", "w"), indent=2)
