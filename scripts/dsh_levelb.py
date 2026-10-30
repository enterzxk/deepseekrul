"""Is the 'levelb' gap significant? M4_racdr vs M0_tf on the same split.

A 2.1 RMSE gap inside one split is larger than anything studied so far, so it is
tested with the same engine-clustered paired bootstrap used elsewhere.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

def per_engine(tag):
    fs = glob.glob(f"experiments/levelb/{tag}/FD002/*/matched_cutoff_predictions.csv")
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f).assign(seed=int(Path(f).parts[-2].split("_s")[-1])) for f in fs])
    d["sq"] = d.signed_error.astype(float) ** 2
    per = d.groupby(["unit_id", "seed"]).sq.mean().reset_index()
    return per.groupby("unit_id").sq.mean().pow(0.5)

a, b = per_engine("M4_racdr"), per_engine("M0_tf")
print("engines:", None if a is None or b is None else len(a.index.intersection(b.index)))
idx = a.index.intersection(b.index)
x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
rng = np.random.default_rng(20261022)
dd = []
for _ in range(5000):
    t = rng.integers(0, len(x), len(x))
    dd.append(np.sqrt((x[t] ** 2).mean()) - np.sqrt((y[t] ** 2).mean()))
dd = np.asarray(dd)
print(f"racdr={np.sqrt((x**2).mean()):.3f} tf={np.sqrt((y**2).mean()):.3f}")
print(f"diff={np.median(dd):.3f} CI=[{np.percentile(dd,2.5):.3f},{np.percentile(dd,97.5):.3f}] "
      f"P(racdr better)={(dd<0).mean():.3f} excludes0={bool(np.percentile(dd,2.5)>0 or np.percentile(dd,97.5)<0)}")
# also check what 'levelb' actually is
import json
p = Path("experiments/levelb/M4_racdr/FD002")
runs = list(p.glob("*/metrics.json"))
if runs:
    m = json.load(open(runs[0]))
    print("levelb keys:", sorted(m)[:12])
    print("split_seed:", m.get("split_seed"), "variant:", m.get("variant"), "val_units:", len(m.get("val_units", [])))
