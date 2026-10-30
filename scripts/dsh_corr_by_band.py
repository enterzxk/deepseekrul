"""Is the 0.98 prediction correlation inflated by the RUL cap?

With cap=125, early windows all have target 125, so every model predicts ~125 and
correlations are trivially high. This recomputes pairwise correlation restricted
to windows whose true RUL is below the cap, where predictions can actually differ.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/dsh/review_r1")

def load_cell(cell):
    fs = sorted(glob.glob(f"experiments/order03/{cell}/FD002/*/protocol_c_predictions.csv"))
    if not fs:
        return None
    ds = [pd.read_csv(f) for f in fs]
    d = ds[0][["unit_id", "abs_cut", "y_true"]].copy()
    d["y_pred"] = np.mean([x.y_pred.to_numpy() for x in ds], axis=0)
    return d

cells = sorted({Path(p).parts[2] for p in glob.glob("experiments/order03/*/FD002/*/protocol_c_predictions.csv")})
frames = {}
for c in cells:
    d = load_cell(c)
    if d is not None:
        frames[c] = d

def corr_table(mask):
    cols = {c: d.loc[mask.reindex(d.index, fill_value=False), "y_pred"].to_numpy() for c, d in frames.items()}
    df = pd.DataFrame(cols)
    return df.corr()

any_d = next(iter(frames.values()))
for label, thr in (("all", 1e9), ("RUL<125", 125), ("RUL<100", 100), ("RUL<60", 60)):
    mask = any_d.y_true < thr
    m = mask.to_numpy()
    df = pd.DataFrame({c: d.y_pred.to_numpy()[m] for c, d in frames.items()})
    if df.shape[1] < 2 or df.shape[0] < 20:
        continue
    cm = df.corr().where(~np.eye(df.shape[1], dtype=bool)).stack()
    print(f"{label:9s} windows={df.shape[0]:5d}  median_r={cm.median():.4f}  min_r={cm.min():.4f}  frac_r<0.9={float((cm<0.9).mean()):.3f}")
