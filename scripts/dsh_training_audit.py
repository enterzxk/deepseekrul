"""Are the models under-trained, or converged early to a poor optimum?

Reads history.json from existing runs: best epoch, total epochs, and whether the
validation curve was still improving at the end. This decides whether "train
longer" is even the right remedy for the poor RMSE levels.
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
    hist = h if isinstance(h, list) else (h.get("history") or [])
    vals = [r.get("val_rmse") for r in hist if isinstance(r, dict) and r.get("val_rmse") is not None]
    vals = [v for v in vals if v is not None]
    if not vals:
        continue
    best_i = int(np.argmin(vals))
    cell = Path(f).parts[-4]
    rows.append({"cell": cell, "epochs": len(vals), "best_epoch": best_i + 1,
                 "best_val": float(min(vals)), "last_val": float(vals[-1]),
                 "improved_in_last_5": bool(min(vals[-5:]) < min(vals[:-5])) if len(vals) > 5 else None})
df = pd.DataFrame(rows)
if df.empty:
    print("no history files parsed; keys may differ")
else:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "training_audit.csv", index=False)
    print(df.groupby("cell").agg(n=("epochs", "size"), mean_epochs=("epochs", "mean"),
                                 mean_best_epoch=("best_epoch", "mean"),
                                 mean_best_val=("best_val", "mean")).round(2).to_string())
    print()
    print(f"runs where best epoch == total epochs (still improving at stop): "
          f"{int((df.best_epoch == df.epochs).sum())} / {len(df)}")
    print(f"median epochs run: {df.epochs.median():.0f}")
