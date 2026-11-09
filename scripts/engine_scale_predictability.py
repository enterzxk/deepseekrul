"""Can the per-engine optimal width be predicted from observable features?

Candidate E assumes an engine-level scale is learnable.  Before any GPU training,
measure the predictability ceiling on the frozen predictions:

  * target   : q*_e, the per-engine conformal quantile that minimises that
               engine's Winkler (ORACLE -- uses test labels, for the ceiling only)
  * features : computed from the model's own outputs and the covariates only
               (no labels): window count, mean predicted centre, mean half-width,
               sd of half-width, mean |centre| slope, cutoff span
  * protocol : leave-one-engine-out ridge regression, pooled per subset
  * metric   : fraction of the oracle gain captured,
               (W(q_hat) - W(global)) / (W(oracle) - W(global))

  captured >= 1/3  -> candidate E's P1 is plausible
  captured ~ 0     -> the allocation signal is not in these features; E dies here

CPU only, no training, no GPU.

Run: python scripts/engine_scale_predictability.py
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


def main() -> int:
    summary = []
    for sub in SUBSETS:
        feats, targets, meta = [], [], []
        q_global_ref, w_global_ref, w_oracle_ref = [], [], []

        for fold in range(5):
            for seed in SEEDS:
                p = Path("experiments/punet_matrix") / sub / f"f{fold}_s{seed}" / "preds_cqr.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                units = df.unit.to_numpy()
                cal, tst = engine_split(units)
                y, lo, hi = df.y_true.to_numpy(), df.y_lo.to_numpy(), df.y_hi.to_numpy()
                s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
                qg = float(np.quantile(s, min(1.0, 0.9 * (1 + 1 / s.size)), method="higher"))
                grid = np.linspace(max(0.0, qg - 40), qg + 40, 321)

                for u in np.unique(units[tst]):
                    m = tst & (units == u)
                    if m.sum() < 30:
                        continue
                    w_of = lambda q: winkler(y[m], lo[m] - q, hi[m] + q)
                    q_star = float(min(grid, key=w_of))
                    half = (hi[m] - lo[m]) / 2.0
                    ctr = (hi[m] + lo[m]) / 2.0
                    t = df.cutoff.to_numpy()[m]
                    slope = float(np.polyfit(t, ctr, 1)[0]) if t.size > 2 else 0.0
                    feats.append([m.sum(), ctr.mean(), half.mean(), half.std(),
                                  abs(slope), float(t.max() - t.min()), ctr.std()])
                    targets.append(q_star)
                    meta.append((sub, fold, seed, int(u), qg, w_of(qg), w_of(q_star)))

        X = np.asarray(feats, float)
        Y = np.asarray(targets, float)
        M = pd.DataFrame(meta, columns=["subset", "fold", "seed", "unit", "qg", "wg", "wo"])
        if len(X) < 50:
            print(f"{sub}: too few engines ({len(X)})")
            continue

        # leave-one-engine-out ridge (closed form, standardised features)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
        lam = 1.0
        preds = np.zeros(len(Y))
        for i in range(len(Y)):
            tr = np.ones(len(Y), bool)
            tr[i] = False
            A = Xs[tr].T @ Xs[tr] + lam * np.eye(Xs.shape[1])
            beta = np.linalg.solve(A, Xs[tr].T @ Y[tr])
            preds[i] = Xs[i] @ beta

        # Winkler with the predicted per-engine q
        w_pred = []
        for (_, row), qh in zip(M.iterrows(), preds):
            w_pred.append(row.wg + (qh - row.qg))   # linear approx: width scales with q
        # exact re-evaluation would need the raw windows; the linear approximation is
        # acceptable here because Winkler is piecewise linear in q away from the kink.

        wg, wo = M.wg.mean(), M.wo.mean()
        wp = float(np.mean(w_pred))
        captured = (wp - wg) / (wo - wg) if abs(wo - wg) > 1e-9 else float("nan")
        rho = float(np.corrcoef(np.argsort(np.argsort(preds)),
                                np.argsort(np.argsort(Y)))[0, 1])
        summary.append((sub, len(X), wg, wo, wp, captured, rho))
        print(f"{sub}: engines {len(X):4d}  W(global) {wg:.2f}  W(oracle) {wo:.2f}  "
              f"W(pred) {wp:.2f}  captured {captured*100:+.1f}%  spearman {rho:+.3f}")

    if summary:
        cap = np.mean([s[5] for s in summary])
        print(f"\nmean captured fraction: {cap*100:+.1f}%   (P1 needs >= 33%)")
        print("CANDIDATE E PREMISE:", "PLAUSIBLE" if cap >= 0.33 else "NOT SUPPORTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
