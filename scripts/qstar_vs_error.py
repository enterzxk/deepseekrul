"""Is the per-engine optimal width just the per-engine error magnitude?

The hand-feature ceiling test (scripts/engine_scale_predictability.py) failed, but
it used crude trajectory statistics.  A learned engine-level scale would be trained
on residuals, so the relevant question is: does the ORACLE optimal q* track the
engine's actual error magnitude?

  high rank correlation  -> the target is "predict your own error", which a
                            residual-trained sigma_eng could in principle learn
  low rank correlation   -> the optimal width is driven by something else
                            (asymmetry, tail shape, label noise) and E is dead

CPU only, frozen predictions.

Run: python scripts/qstar_vs_error.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 7, 2024, 13]


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, a=0.1):
    return float(np.mean((hi - lo) + (2.0 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def main() -> int:
    all_q, all_err, all_abs, all_asym = [], [], [], []
    for sub in SUBSETS:
        qs, errs, abss, asyms = [], [], [], []
        for fold in range(5):
            for seed in SEEDS:
                p = Path("experiments/punet_matrix") / sub / f"f{fold}_s{seed}" / "preds_cqr.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                units = df.unit.to_numpy()
                cal, tst = engine_split(units)
                y, lo, hi = df.y_true.to_numpy(), df.y_lo.to_numpy(), df.y_hi.to_numpy()
                ctr = (hi + lo) / 2.0
                s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
                qg = float(np.quantile(s, min(1.0, 0.9 * (1 + 1 / s.size)), method="higher"))
                grid = np.linspace(max(0.0, qg - 40), qg + 40, 321)
                for u in np.unique(units[tst]):
                    m = tst & (units == u)
                    if m.sum() < 30:
                        continue
                    q_star = float(min(grid, key=lambda q: winkler(y[m], lo[m] - q, hi[m] + q)))
                    r = y[m] - ctr[m]
                    qs.append(q_star)
                    errs.append(float(np.sqrt(np.mean(r ** 2))))
                    abss.append(float(np.mean(np.abs(r))))
                    # asymmetry of the residual distribution
                    asyms.append(float(np.mean(r) / (np.mean(np.abs(r)) + 1e-9)))
        if qs:
            print(f"{sub}: engines {len(qs):4d}  "
                  f"rho(q*, RMSE) {spearman(qs, errs):+.3f}  "
                  f"rho(q*, MAE) {spearman(qs, abss):+.3f}  "
                  f"rho(q*, asym) {spearman(qs, asyms):+.3f}")
            all_q += qs
            all_err += errs
            all_abs += abss
            all_asym += asyms

    print(f"\nALL: engines {len(all_q)}  rho(q*, RMSE) {spearman(all_q, all_err):+.3f}  "
          f"rho(q*, MAE) {spearman(all_q, all_abs):+.3f}")
    r = spearman(all_q, all_err)
    print("INTERPRETATION:", "TARGET IS LEARNABLE IN PRINCIPLE" if r >= 0.5 else
          ("PARTIALLY LEARNABLE" if r >= 0.3 else "NOT EXPLAINED BY ERROR MAGNITUDE"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
