"""Fair comparison on a common yardstick (standard capped RUL).

CP-labelled models predict min(RUL, plateau); baselines predict min(RUL, 125).
Scoring each on its own target is meaningless, so both predictions are scored
against the SAME truth: the baseline's y_true (standard capped RUL), aligned by
unit_id (one matched cutoff per engine).
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/dsh/review_r1")

def load(tag):
    fs = glob.glob(f"experiments/{tag}/FD002/*/matched_cutoff_predictions.csv")
    if not fs:
        return None
    return pd.concat([pd.read_csv(f).assign(model_seed=int(Path(f).parts[-2].split("_s")[-1])) for f in fs])

base = load("order03/M1_global")
cp = load("cplab2")
truth = base.groupby("unit_id").y_true.mean().rename("rul_true")
print("engines with truth:", len(truth))

rows = []
for name, d in (("M1_global (cap labels)", base), ("cplab2 (change-point labels)", cp)):
    g = d.groupby("unit_id").y_pred.mean().rename("pred").to_frame().join(truth, how="inner")
    e = (g.pred - g.rul_true).to_numpy()
    rows.append({"model": name, "engines": len(g),
                 "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
                 "late_frac": float((e > 0).mean()), "severe_gt10": int((e > 10).sum()),
                 "bias": float(e.mean())})
out = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT / "cp_vs_baseline_common_truth.csv", index=False)
print(out.round(3).to_string(index=False))

# prediction agreement, the distinguishability check that matters
p1 = base.groupby("unit_id").y_pred.mean()
p2 = cp.groupby("unit_id").y_pred.mean()
j = pd.concat([p1.rename("m1"), p2.rename("cp")], axis=1).dropna()
print()
print("prediction correlation:", round(float(j.corr().iloc[0, 1]), 4))
