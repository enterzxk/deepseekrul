"""Are the methods actually different? Pairwise correlation of predictions.

If two methods' predictions correlate above ~0.99 on the same windows, they are
behaviourally the same model on this data, and any RMSE difference is numerical
noise rather than a methodological effect.
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/dsh/review_r1")
series = {}
for f in sorted(glob.glob("experiments/order03/*/FD002/*/protocol_c_predictions.csv")):
    cell = Path(f).parts[-4]
    d = pd.read_csv(f).sort_values(["unit_id", "abs_cut"])
    key = d.unit_id.astype(str) + "_" + d.abs_cut.astype(str)
    series.setdefault(cell, []).append(pd.Series(d.y_pred.to_numpy(), index=key))

# average over seeds where a cell has several
mat = {}
for cell, lst in series.items():
    keys = lst[0].index if len(lst) == 1 else lst[0].index
    mat[cell] = np.mean([s.reindex(keys).to_numpy() for s in lst], axis=0)
df = pd.DataFrame(mat)
corr = df.corr()
OUT.mkdir(parents=True, exist_ok=True)
corr.round(4).to_csv(OUT / "prediction_corr.csv")
print(corr.round(3).to_string())
print()
pairs = corr.where(~np.eye(len(corr), dtype=bool)).stack().sort_values()
print("lowest pairwise correlations:")
print(pairs.head(6).round(4).to_string())
print()
print(f"median pairwise correlation: {pairs.median():.4f}")
print(f"pairs with r < 0.99: {int((pairs < 0.99).sum())} / {len(pairs)}")
