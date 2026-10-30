"""Variance decomposition of all performance data available.

Design: 3 methods x 3 engine splits x 3 model seeds = 27 matched-RMSE observations
per subset. A two-way ANOVA (method, split) with interaction separates how much of
the variation is attributable to the method choice versus the split choice.
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
    d["sq"] = d.signed_error.astype(float) ** 2
    per = d.groupby(["unit_id", "model_seed"]).sq.mean().reset_index()
    return per.groupby("model_seed").sq.mean().pow(0.5)

rows = []
for name, root in ROOTS.items():
    for tag in TAGS:
        r = rmse(root, tag)
        if r is None:
            continue
        for seed, v in r.items():
            rows.append({"split": name, "method": tag, "seed": int(seed), "rmse": float(v)})
df = pd.DataFrame(rows)
df.to_csv(OUT / "perf_design.csv", index=False)
print(df.pivot_table(index="method", columns="split", values="rmse", aggfunc="mean").round(2).to_string())

y = df.rmse.to_numpy()
grand = y.mean()
ss_total = ((y - grand) ** 2).sum()
def ss_between(col):
    g = df.groupby(col).rmse
    return float((g.count() * (g.mean() - grand) ** 2).sum())
ss_method = ss_between("method")
ss_split = ss_between("split")
cell = df.groupby(["method", "split"]).rmse
ss_cell = float((cell.count() * (cell.mean() - grand) ** 2).sum())
ss_inter = max(0.0, ss_cell - ss_method - ss_split)
ss_error = ss_total - ss_cell
out = {
    "n": len(df), "grand_mean": float(grand),
    "pct_method": 100 * ss_method / ss_total,
    "pct_split": 100 * ss_split / ss_total,
    "pct_interaction": 100 * ss_inter / ss_total,
    "pct_residual_seed": 100 * ss_error / ss_total,
    "sd_method_means": float(df.groupby("method").rmse.mean().std(ddof=1)),
    "sd_split_means": float(df.groupby("split").rmse.mean().std(ddof=1)),
}
json.dump(out, open(OUT / "variance_decomposition.json", "w"), indent=2)
print()
print(json.dumps(out, indent=2))
