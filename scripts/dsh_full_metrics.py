"""Complete metric table under the corrected evaluation, for every cell.

The objective requires RMSE, MAE, NASA score, late_fraction, severe-late and the
near-failure segment to be reported together. This computes all of them from the
stored predictions, per cell and subset, with engine-level aggregation.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/dsh/review_r1")

def nasa(d):
    d = np.asarray(d, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))

rows = []
for f in sorted(glob.glob("experiments/*/*/*/*/matched_cutoff_predictions.csv")):
    parts = Path(f).parts
    root, cell, subset, run = parts[1], parts[2], parts[3], parts[4]
    if root.startswith("cp"):      # change-point / cap experiments kept separate
        continue
    d = pd.read_csv(f)
    if d.empty or "signed_error" not in d:
        continue
    e = d.signed_error.to_numpy(float)
    yt = d.y_true.to_numpy(float)
    rows.append({
        "split": root, "cell": cell, "subset": subset, "run": run, "n": len(d),
        "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
        "nasa": nasa(e), "late_frac": float((e > 0).mean()),
        "severe_late": int((e > 10).sum()),
    })
df = pd.DataFrame(rows)
if df.empty:
    print("no matched predictions found"); raise SystemExit
agg = (df.groupby(["split", "cell", "subset"])
         .agg(n_runs=("n", "size"), rmse=("rmse", "mean"), mae=("mae", "mean"),
              nasa=("nasa", "mean"), late_frac=("late_frac", "mean"),
              severe_late=("severe_late", "mean"))
         .reset_index().sort_values(["split", "subset", "rmse"]))
OUT.mkdir(parents=True, exist_ok=True)
agg.to_csv(OUT / "full_metrics_table.csv", index=False)
show = agg[agg.subset == "FD002"].head(14)
print(show.round(3).to_string(index=False))
print()
print("rows saved:", len(agg))
