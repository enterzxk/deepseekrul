"""Variance decomposition of the evaluation itself.

The same method evaluated on three engine splits gives RMSE 12.93 / 15.57 / 17.22
on FD002 — a spread far larger than the between-method differences usually
claimed. This quantifies, for each method, the split-to-split spread, and compares
it with the within-split between-method differences.
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOTS = {"split7": "experiments/order03", "split8": "experiments/split8", "split9": "experiments/split9"}
TAGS = ["M0_raw_tf", "M1_global", "M6_tfA"]
OUT = Path("results/dsh/review_r1")

def rmse(root, tag, subset="FD002"):
    fs = glob.glob(f"{root}/{tag}/{subset}/*/matched_cutoff_predictions.csv")
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f).assign(model_seed=int(Path(f).parts[-2].split("_s")[-1])) for f in fs])
    sq = (d.signed_error.astype(float) ** 2)
    per = d.assign(sq=sq).groupby(["unit_id", "model_seed"]).sq.mean().reset_index()
    return float(np.sqrt(per.groupby("unit_id").sq.mean().mean()))

rows = []
for tag in TAGS:
    vals = {name: rmse(root, tag) for name, root in ROOTS.items()}
    vals = {k: v for k, v in vals.items() if v is not None}
    if len(vals) < 3:
        continue
    v = np.array(list(vals.values()))
    rows.append({"method": tag, **{f"rmse_{k}": round(x, 3) for k, x in vals.items()},
                 "mean": float(v.mean()), "sd_across_splits": float(v.std(ddof=1)),
                 "range": float(v.max() - v.min())})
df = pd.DataFrame(rows)
df.to_csv(OUT / "split_variance.csv", index=False)
print(df.round(3).to_string(index=False))
print()
print(f"median split-to-split sd (same method): {df.sd_across_splits.median():.2f}")
print(f"median split-to-split range        : {df.range.median():.2f}")
# within-split between-method spread, for comparison
for name, root in ROOTS.items():
    vals = [rmse(root, t) for t in TAGS]
    vals = [v for v in vals if v is not None]
    if len(vals) >= 2:
        print(f"within {name}: between-method range = {max(vals)-min(vals):.2f}")
