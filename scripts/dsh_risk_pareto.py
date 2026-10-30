"""Pareto sweep: accuracy vs severe-late under asymmetric risk calibration.

For each weight w_late, fit the offset on half the engines and evaluate on the
other half; report RMSE and severe-late. This traces the trade-off curve that a
deployment could choose from.

Usage: python scripts/dsh_risk_pareto.py --root experiments/order03
"""
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd


def load(root, tag, subset):
    fs = glob.glob(str(Path(root) / tag / subset / "*" / "matched_cutoff_predictions.csv"))
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else None


def fit(e, w, grid=np.arange(-15, 15.01, 0.25)):
    best, cost = 0.0, np.inf
    for d in grid:
        r = e - d
        c = np.where(r > 0, w * r ** 2, r ** 2).mean()
        if c < cost:
            best, cost = float(d), c
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/order03")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    tags = ["M0_raw_tf", "M6_tfA", "M1_global", "M4b_popgate", "M5_globalgate"]
    weights = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    rng = np.random.default_rng(args.seed)
    rows = []
    for subset in ("FD001", "FD002"):
        for tag in tags:
            df = load(args.root, tag, subset)
            if df is None:
                continue
            units = np.array(sorted(df.unit_id.unique()))
            for w in weights:
                rec = []
                for _ in range(args.reps):
                    p = rng.permutation(units)
                    fit_u, ev_u = p[: len(p) // 2], p[len(p) // 2:]
                    d = fit(df[df.unit_id.isin(fit_u)].signed_error.to_numpy(), w)
                    e = df[df.unit_id.isin(ev_u)].signed_error.to_numpy() - d
                    rec.append((np.sqrt((e ** 2).mean()), (e > 0).mean(), (e > 10).sum()))
                a = np.array(rec)
                rows.append({"subset": subset, "cell": tag, "w_late": w,
                             "rmse": a[:, 0].mean(), "late": a[:, 1].mean(), "severe": a[:, 2].mean()})
    out = pd.DataFrame(rows)
    Path("results/dsh").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/dsh/risk_pareto.csv", index=False)
    for subset in ("FD001", "FD002"):
        print(f"=== {subset} ===")
        piv = out[out.subset == subset].pivot(index="cell", columns="w_late", values=["rmse", "severe"])
        print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
