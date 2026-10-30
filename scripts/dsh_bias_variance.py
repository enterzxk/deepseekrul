"""Bias vs variance: compare final training error with validation error.

If sqrt(train_loss) is close to val RMSE, the models are underfitting (high bias);
if train is much lower, they are overfitting. This decides whether the remedy is
more capacity/features or better regularisation/optimisation.
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/dsh/review_r1")
rows = []
for f in glob.glob("experiments/order03/*/FD002/*/history.json"):
    try:
        h = json.load(open(f))
    except Exception:
        continue
    hist = h if isinstance(h, list) else h.get("history") or []
    pts = [r for r in hist if isinstance(r, dict) and r.get("val_rmse") is not None]
    if not pts:
        continue
    best = min(pts, key=lambda r: r["val_rmse"])
    rows.append({"cell": Path(f).parts[-4], "seed": Path(f).parts[-2],
                 "train_rmse_at_best": float(np.sqrt(max(best.get("train_loss", np.nan), 0))),
                 "val_rmse_at_best": float(best["val_rmse"]),
                 "train_rmse_final": float(np.sqrt(max(pts[-1].get("train_loss", np.nan), 0))),
                 "val_rmse_final": float(pts[-1]["val_rmse"])})
df = pd.DataFrame(rows)
df["gap"] = df.train_rmse_at_best - df.val_rmse_at_best
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "bias_variance.csv", index=False)
g = df.groupby("cell").agg(n=("gap", "size"),
                           train=("train_rmse_at_best", "mean"),
                           val=("val_rmse_at_best", "mean"),
                           gap=("gap", "mean")).round(2)
print(g.to_string())
print()
print(f"median train-val gap: {df.gap.median():.2f}")
print(f"runs where train RMSE > val RMSE (clear underfit): {int((df.gap > 0.5).sum())} / {len(df)}")
print(f"runs where train RMSE much < val (overfit): {int((df.gap < -0.5).sum())} / {len(df)}")
