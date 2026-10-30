"""Re-check the three-split effect with explicit keys and engine clustering.

The earlier effect_across_splits.csv came from a loader that keyed errors by
(subset, cell, unit) -> {cutoff: err}, the same shape that silently dropped seeds
elsewhere. This redoes it from matched-cutoff predictions with explicit
(unit, seed) records, per-engine RMSE, and an engine-clustered paired bootstrap.
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOTS = {"split7": "experiments/order03", "split8": "experiments/split8", "split9": "experiments/split9"}
OUT = Path("results/dsh/review_r1")

def per_engine(root, tag, subset="FD002"):
    fs = glob.glob(f"{root}/{tag}/{subset}/*/matched_cutoff_predictions.csv")
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f).assign(model_seed=int(Path(f).parts[-2].split("_s")[-1])) for f in fs])
    d["sq"] = d.signed_error.astype(float) ** 2
    per = d.groupby(["unit_id", "model_seed"]).sq.mean().reset_index()
    return per.groupby("unit_id").sq.mean().pow(0.5)

def boot(a, b, reps=5000, seed=20261022):
    idx = a.index.intersection(b.index)
    if len(idx) < 5:
        return None
    x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(reps):
        t = rng.integers(0, len(idx), len(idx))
        d.append(np.sqrt((x[t] ** 2).mean()) - np.sqrt((y[t] ** 2).mean()))
    d = np.asarray(d)
    return {"engines": int(len(idx)), "a_rmse": float(np.sqrt((x ** 2).mean())),
            "b_rmse": float(np.sqrt((y ** 2).mean())), "median_diff": float(np.median(d)),
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "prob_a_better": float((d < 0).mean()),
            "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}

rows = []
for name, root in ROOTS.items():
    a, b = per_engine(root, "M1_global"), per_engine(root, "M6_tfA")
    if a is None or b is None:
        print(name, "missing"); continue
    r = boot(a, b)
    r.update({"split": name, "pair": "M1 vs M6"})
    rows.append(r)
    print(json.dumps(r))
df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "three_split_recheck.csv", index=False)
print()
print("excludes zero in all splits:", bool(df.excludes_zero.all()))
print("median diffs:", df.median_diff.round(3).tolist())
