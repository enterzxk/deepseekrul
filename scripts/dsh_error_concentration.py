"""Error concentration: how much of the total squared error comes from a few engines?

If a small subset of engines dominates the MSE, then any method difference is
decided on ~5-10 engines regardless of how many are in the validation set, which
bounds the achievable significance.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/dsh/review_r1")

def per_engine(cell):
    fs = glob.glob(f"experiments/order03/{cell}/FD002/*/protocol_c_predictions.csv")
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d["sq"] = d.signed_error.astype(float) ** 2
    return d.groupby("unit_id").sq.mean()

rows = []
err_series = {}
for cell in sorted({Path(p).parts[2] for p in glob.glob("experiments/order03/*/FD002/*/protocol_c_predictions.csv")}):
    s = per_engine(cell)
    if s is None:
        continue
    err_series[cell] = s
    tot = s.sum()
    top10 = s.nlargest(max(1, int(round(0.1 * len(s))))).sum()
    rows.append({"cell": cell, "engines": len(s),
                 "top10pct_share": float(top10 / tot),
                 "share_of_worst_5": float(s.nlargest(5).sum() / tot),
                 "median_engine_mse": float(s.median()),
                 "max_engine_mse": float(s.max())})
df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "error_concentration.csv", index=False)
print(df.round(3).to_string(index=False))
print()

E = pd.DataFrame(err_series)
c = E.corr()
print("correlation of per-engine squared error across methods:")
print(c.where(~np.eye(len(c), dtype=bool)).stack().describe().round(3).to_string())
print()
# what would happen if only the worst engines decide the ranking
print("median share of MSE from the worst 10% of engines:",
      round(float(df.top10pct_share.median()), 3))
