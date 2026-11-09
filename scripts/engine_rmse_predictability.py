"""Can an engine's error magnitude be predicted from the model's own outputs?

Candidate E's engine-level head sees a pooled trajectory representation.  As a
CPU-only proxy, build a richer per-engine feature vector from the model's own
outputs (no labels) and check how well it predicts the engine's RMSE with
leave-one-engine-out ridge regression.

  rho >= 0.60  -> the information is in the outputs; the sigma_eng head has a
                  target it can plausibly learn, and a 20 GPU-h pilot is justified
  rho <  0.45  -> the pooled head is unlikely to beat the per-window half-width

CPU only, frozen predictions, no training.

Run: python scripts/engine_rmse_predictability.py
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


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def features(t, ctr, half):
    """Per-engine features from the model's outputs and the covariates only."""
    n = len(t)
    dt = np.diff(t) if n > 1 else np.array([1.0])
    sl_c = np.polyfit(t, ctr, 1)[0] if n > 2 else 0.0
    sl_h = np.polyfit(t, half, 1)[0] if n > 2 else 0.0
    # curvature of the predicted centre (second difference)
    cur = float(np.mean(np.abs(np.diff(ctr, 2)))) if n > 3 else 0.0
    return [
        n, float(t.max() - t.min()), float(np.mean(dt)), float(np.std(dt)),
        float(ctr.mean()), float(ctr.std()), float(ctr.min()), float(ctr.max()),
        sl_c, float(np.std(np.diff(ctr))) if n > 1 else 0.0, cur,
        float(half.mean()), float(half.std()), float(half.min()), float(half.max()),
        sl_h, float(half.max() / (half.min() + 1e-6)),
        float(np.corrcoef(t, half)[0, 1]) if n > 2 and half.std() > 1e-9 else 0.0,
        float(np.corrcoef(ctr, half)[0, 1]) if n > 2 and half.std() > 1e-9 and ctr.std() > 1e-9 else 0.0,
        float(np.mean(np.abs(np.diff(ctr)))) if n > 1 else 0.0,
    ]


def main() -> int:
    summary = []
    for sub in SUBSETS:
        X, y = [], []
        for fold in range(5):
            for seed in SEEDS:
                p = Path("experiments/punet_matrix") / sub / f"f{fold}_s{seed}" / "preds_cqr.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                units = df.unit.to_numpy()
                _, tst = engine_split(units)
                yv, lo, hi = df.y_true.to_numpy(), df.y_lo.to_numpy(), df.y_hi.to_numpy()
                ctr = (hi + lo) / 2.0
                t = df.cutoff.to_numpy()
                for u in np.unique(units[tst]):
                    m = tst & (units == u)
                    if m.sum() < 30:
                        continue
                    X.append(features(t[m], ctr[m], (hi[m] - lo[m]) / 2.0))
                    y.append(float(np.sqrt(np.mean((yv[m] - ctr[m]) ** 2))))

        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if len(X) < 100:
            continue
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
        lam = 3.0
        pred = np.zeros(len(y))
        for i in range(len(y)):
            tr = np.ones(len(y), bool)
            tr[i] = False
            A = Xs[tr].T @ Xs[tr] + lam * np.eye(Xs.shape[1])
            beta = np.linalg.solve(A, Xs[tr].T @ y[tr])
            pred[i] = Xs[i] @ beta
        rho = spearman(pred, y)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        summary.append(rho)
        print(f"{sub}: engines {len(y):4d}  rho(pred RMSE, true RMSE) {rho:+.3f}  R2 {r2:+.3f}")

    if summary:
        m = float(np.mean(summary))
        print(f"\nmean rho {m:+.3f}   (E needs >= 0.60 to justify the pilot)")
        print("CANDIDATE E SIGNAL:", "IN THE OUTPUTS" if m >= 0.60 else
              ("WEAK" if m >= 0.45 else "NOT IN THE OUTPUTS"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
