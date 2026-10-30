"""Did the models collapse to near-constant predictions?

A model that predicts ~the mean will have small prediction spread and low
correlation with the target, yet can still show a moderate RMSE. This checks the
spread of predictions and their correlation with the true RUL for every cell.
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/dsh/review_r1")
rows = []
for f in glob.glob("experiments/order03/*/FD002/*/protocol_c_predictions.csv"):
    d = pd.read_csv(f)
    if d.empty:
        continue
    y, p = d.y_true.to_numpy(float), d.y_pred.to_numpy(float)
    rows.append({"cell": Path(f).parts[-4],
                 "pred_sd": float(np.std(p)),
                 "true_sd": float(np.std(y)),
                 "ratio_sd": float(np.std(p) / max(np.std(y), 1e-9)),
                 "corr": float(np.corrcoef(y, p)[0, 1]),
                 "pred_min": float(p.min()), "pred_max": float(p.max()),
                 "rmse": float(np.sqrt(((p - y) ** 2).mean()))})
df = pd.DataFrame(rows)
g = df.groupby("cell").agg(n=("corr", "size"), pred_sd=("pred_sd", "mean"),
                           true_sd=("true_sd", "mean"), ratio=("ratio_sd", "mean"),
                           corr=("corr", "mean"), rmse=("rmse", "mean")).round(3)
OUT.mkdir(parents=True, exist_ok=True)
g.to_csv(OUT / "prediction_collapse.csv")
print(g.to_string())
print()
print(f"cells with prediction/target sd ratio < 0.5: {int((g.ratio < 0.5).sum())} / {len(g)}")
print(f"cells with correlation < 0.5: {int((g['corr'] < 0.5).sum())} / {len(g)}")
