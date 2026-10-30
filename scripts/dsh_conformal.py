"""Split-conformal intervals for RUL, evaluated with engine-level folds.

Definition: for a nominal level 1-alpha, calibrate the absolute-residual quantile on
held-out engines and check empirical coverage on the remaining engines. This tests
whether the point model's errors support a distribution-free guarantee, which the
objective asks for explicitly (coverage/error by an explicit statistical definition).
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/dsh/review_r1")
fs = glob.glob("experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv")
d = pd.concat([pd.read_csv(f).assign(seed=int(Path(f).parts[-2].split("_s")[-1])) for f in fs])
g = d.groupby("unit_id").agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
resid = np.abs(g.y_pred - g.y_true).to_numpy()
units = g.index.to_numpy()

rng = np.random.default_rng(20261022)
rows = []
for alpha in (0.2, 0.1):
    covs, widths = [], []
    for _ in range(200):
        idx = rng.permutation(len(units))
        half = len(units) // 2
        cal, ev = idx[:half], idx[half:]
        # split-conformal quantile with the finite-sample correction:
        # take the ceil((n+1)(1-alpha))-th smallest residual
        ncal = len(cal)
        k = int(np.ceil((ncal + 1) * (1 - alpha)))
        k = min(k, ncal)
        q = np.sort(resid[cal])[k - 1]
        covered = (resid[ev] <= q).mean()
        covs.append(covered); widths.append(2 * q)
    rows.append({"nominal": 1 - alpha, "mean_coverage": float(np.mean(covs)),
                 "sd_coverage": float(np.std(covs)), "mean_width": float(np.mean(widths))})
out = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT / "conformal_coverage.csv", index=False)
print(out.round(3).to_string(index=False))
print()
print(f"engines={len(units)} | residual mean={resid.mean():.2f} sd={resid.std():.2f} "
      f"p90={np.quantile(resid,0.9):.2f} max={resid.max():.1f}")
