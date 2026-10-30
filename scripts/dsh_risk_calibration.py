"""Risk-calibrated RUL post-processing (engine-split, no leakage).

Question: can a single asymmetric offset fitted on half of the engines reduce
severe-late calls on the other half WITHOUT degrading RMSE?

Fit objective (on fit engines): minimise  sum w(e) e^2  with w=2 for e>0 (late).
Evaluation: held-out engines, RMSE / late_fraction / severe-late, bootstrap over engines.

Usage: python scripts/dsh_risk_calibration.py --root experiments/order03
"""
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd


def load_matched(root, tag, subset):
    fs = glob.glob(str(Path(root) / tag / subset / "*" / "matched_cutoff_predictions.csv"))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs])


def metrics(e):
    e = np.asarray(e, dtype=float)
    return {"rmse": float(np.sqrt((e ** 2).mean())),
            "late": float((e > 0).mean()),
            "severe": int((e > 10).sum())}


def fit_offset(e, grid=np.arange(-12, 12.01, 0.25), w_late=2.0):
    best, best_c = 0.0, np.inf
    for d in grid:
        r = e - d
        cost = np.where(r > 0, w_late * r ** 2, r ** 2).mean()
        if cost < best_c:
            best, best_c = float(d), cost
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/order03")
    ap.add_argument("--subset", default="FD002")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    tags = ["M0_raw_tf", "M6_tfA", "M1_global", "M4b_popgate", "M4_racdr", "M5_globalgate"]
    rng = np.random.default_rng(args.seed)
    rows = []
    for tag in tags:
        df = load_matched(args.root, tag, args.subset)
        if df is None:
            continue
        units = np.array(sorted(df.unit_id.unique()))
        base, cal = [], []
        for _ in range(200):
            perm = rng.permutation(units)
            fit, ev = perm[: len(perm) // 2], perm[len(perm) // 2:]
            ef = df[df.unit_id.isin(fit)].signed_error.to_numpy()
            ev_e = df[df.unit_id.isin(ev)].signed_error.to_numpy()
            d = fit_offset(ef)
            base.append(metrics(ev_e))
            cal.append(metrics(ev_e - d))
        b = {k: float(np.mean([x[k] for x in base])) for k in base[0]}
        c = {k: float(np.mean([x[k] for x in cal])) for k in cal[0]}
        rows.append({"cell": tag, "subset": args.subset,
                     "rmse_base": b["rmse"], "rmse_cal": c["rmse"],
                     "late_base": b["late"], "late_cal": c["late"],
                     "severe_base": b["severe"], "severe_cal": c["severe"],
                     "d_rmse_pct": 100 * (c["rmse"] - b["rmse"]) / b["rmse"]})
    out = pd.DataFrame(rows)
    Path("results/dsh").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/dsh/risk_calibration.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
