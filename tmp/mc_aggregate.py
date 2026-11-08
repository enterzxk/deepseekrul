"""Aggregate measurement-condition results (clean / mc1 / mc2) into a table."""
import json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
CONDS = ["clean", "mc1", "mc2"]


def nasa(e):
    e = np.asarray(e, float)
    return np.where(e > 0, np.expm1(e / 10.0), np.expm1(-e / 13.0))


def load(subset, arm, seed, cond):
    blocks = []
    for k in range(5):
        p = HERE / "measurement" / subset / str(k) / f"{arm}_{seed}_{cond}.csv"
        if not p.is_file():
            return None
        blocks.append(pd.read_csv(p))
    d = pd.concat(blocks, ignore_index=True)
    d["err"] = d.y_pred - d.y_true
    return d


def stats(d):
    g = d.groupby("unit")
    t = pd.DataFrame({"mse": g.apply(lambda x: float((x.err ** 2).mean()), include_groups=False),
                      "mae": g.apply(lambda x: float(x.err.abs().mean()), include_groups=False),
                      "nasa": g.apply(lambda x: float(nasa(x.err).mean()), include_groups=False),
                      "bias": g.err.mean()})
    return dict(rmse=float(np.sqrt(t.mse.mean())), mae=float(t.mae.mean()),
                nasa=float(t.nasa.mean()), bias=float(t.bias.mean()), engines=int(len(t)))


rows = []
for subset in ["FD002", "FD004"]:
    for arm in ["M0", "FULL", "NORM"]:
        for cond in CONDS:
            d = load(subset, arm, 101, cond)
            if d is None:
                continue
            rows.append(dict(subset=subset, arm=arm, condition=cond, **stats(d)))
frame = pd.DataFrame(rows)
if len(frame):
    frame.to_csv(HERE / "MEASUREMENT_CONDITIONS.csv", index=False)
    (HERE / "MEASUREMENT_CONDITIONS.json").write_text(frame.to_json(orient="records", indent=2))
    print(frame.round(3).to_string(index=False))
else:
    print("no measurement-condition results yet")
