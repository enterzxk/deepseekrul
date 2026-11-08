"""Lock the per-subset reference arm from seed-101 full-budget outer validation.

Rule (PROTOCOL.md section 4): reference = lowest pooled engine-equal-weight RMSE among
{M0, OPS, FULL} at seed 101; ties prefer fewer extra inputs (M0 > OPS > FULL).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = ["M0", "OPS", "FULL"]
SUBSETS = ["FD002", "FD004"]


def pooled(base, subset, arm, seed=101):
    blocks = []
    for k in range(5):
        p = Path(base) / subset / str(k) / f"{arm}_{seed}.csv"
        if not p.is_file():
            return None
        blocks.append(pd.read_csv(p))
    d = pd.concat(blocks, ignore_index=True)
    e = d.y_pred - d.y_true
    g = pd.DataFrame({"u": d.unit, "e2": e ** 2}).groupby("u").e2.mean()
    return float(np.sqrt(g.mean())), int(len(g)), int(len(d))


out = {"rule": "lowest pooled engine-equal-weight RMSE among M0/OPS/FULL at seed 101, "
               "full budget (80 epochs, all windows, fixed final epoch); ties -> M0 > OPS > FULL",
       "locked_at": "2026-11-05", "subsets": {}}
for subset in SUBSETS:
    vals = {}
    for arm in ARMS:
        r = pooled("runs", subset, arm)
        if r is not None:
            vals[arm] = r
    if not vals:
        continue
    best = min(vals, key=lambda a: (vals[a][0], ARMS.index(a)))
    out["subsets"][subset] = {"values": {a: {"rmse": v[0], "engines": v[1], "windows": v[2]}
                                         for a, v in vals.items()},
                              "reference": best,
                              "reference_rmse": vals[best][0]}
    print(subset, {a: round(v[0], 3) for a, v in vals.items()}, "->", best)
Path("REFERENCE_LOCK.json").write_text(json.dumps(out, indent=2))
