"""Stage-conditional calibration using only the observable observation stage.

Bias flips sign between short and medium observation windows for every model
tested (negative at cycle 60, positive at 100-140). Because the current cycle is
observable at deployment, a per-stage offset is admissible. Cross-fitted on
engines: fit offsets on half the engines, evaluate on the other half.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = "experiments/order03"
CELLS = ["M0_raw_tf", "M1_global", "M6_tfA", "M4_racdr"]
OUT = Path("results/dsh/review_r1")

def load(cell, subset="FD002"):
    fs = glob.glob(f"{ROOT}/{cell}/{subset}/*/protocol_c_predictions.csv")
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs])

def metrics(e):
    e = np.asarray(e, float)
    return dict(rmse=float(np.sqrt((e ** 2).mean())), late=float((e > 0).mean()),
                severe=int((e > 10).sum()), mae=float(np.abs(e).mean()))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261022)
    rows = []
    for cell in CELLS:
        d = load(cell)
        if d is None:
            continue
        units = np.array(sorted(d.unit_id.unique()))
        base, adj = [], []
        for _ in range(200):
            p = rng.permutation(units)
            fit_u, ev_u = p[: len(p) // 2], p[len(p) // 2:]
            f = d[d.unit_id.isin(fit_u)]
            off = f.groupby("abs_cut").signed_error.mean()      # per-stage offset
            ev = d[d.unit_id.isin(ev_u)].copy()
            ev["ecorr"] = ev.signed_error - ev.abs_cut.map(off).fillna(0.0)
            base.append(metrics(ev.signed_error))
            adj.append(metrics(ev["ecorr"]))
        b = {k: float(np.mean([x[k] for x in base])) for k in base[0]}
        a = {k: float(np.mean([x[k] for x in adj])) for k in adj[0]}
        rows.append({"cell": cell,
                     "rmse_base": b["rmse"], "rmse_cal": a["rmse"], "d_rmse_pct": 100 * (a["rmse"] - b["rmse"]) / b["rmse"],
                     "late_base": b["late"], "late_cal": a["late"],
                     "severe_base": b["severe"], "severe_cal": a["severe"],
                     "mae_base": b["mae"], "mae_cal": a["mae"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stage_calibration.csv", index=False)
    print(df.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
