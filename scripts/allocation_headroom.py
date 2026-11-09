"""Is there headroom in per-engine width allocation?

Every device-level method we tried (A, C, G1) ends up "more correct but wider".
Before designing another one, measure the ORACLE headroom: if we could choose a
separate conformal quantile q_e for every test engine (cheating, using test
labels), how much Winkler would we save relative to one global q?

  * large oracle gain  -> the allocation problem is worth solving
  * small oracle gain  -> no device-level allocation method can win on Winkler,
                          and the "device-level" framing is a dead end for
                          performance (it may still be a validity contribution)

CPU only, frozen predictions, no training.

Run: python scripts/allocation_headroom.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
DELTAS = np.round(np.arange(0.0, 0.61, 0.01), 2)


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, a=0.1):
    return float(np.mean((hi - lo) + (2.0 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def main() -> int:
    rows = []
    for sub in SUBSETS:
        for fold in range(5):
            p = Path("experiments/punet_matrix") / sub / f"f{fold}_s42" / "preds_cqr.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            units = df.unit.to_numpy()
            cal, tst = engine_split(units)
            y, lo, hi = df.y_true.to_numpy(), df.y_lo.to_numpy(), df.y_hi.to_numpy()

            # global conformal quantile on the calibration engines (the frozen protocol)
            s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
            q_global = float(np.quantile(s, min(1.0, 0.9 * (1 + 1 / s.size)), method="higher"))
            w_global = winkler(y[tst], lo[tst] - q_global, hi[tst] + q_global)

            # oracle: one q per test engine, chosen to minimise that engine's Winkler
            w_oracle_parts, n_win = [], 0
            for u in np.unique(units[tst]):
                m = tst & (units == u)
                best = min(winkler(y[m], lo[m] - q, hi[m] + q) for q in
                           np.quantile(s, np.clip(0.5 + DELTAS / 2, 0, 1)) if True)
                # search a fine grid of absolute q values around the global one
                grid = np.linspace(max(0.0, q_global - 30), q_global + 30, 241)
                best = min(winkler(y[m], lo[m] - q, hi[m] + q) for q in grid)
                w_oracle_parts.append(best * m.sum())
                n_win += m.sum()
            w_oracle = float(np.sum(w_oracle_parts) / n_win)

            rows.append(dict(subset=sub, fold=fold, w_global=w_global, w_oracle=w_oracle,
                             gain_pct=(w_oracle - w_global) / w_global * 100))

    d = pd.DataFrame(rows)
    if d.empty:
        print("no cells")
        return 1
    print(f"{'subset':<8}{'Winkler global':>15}{'Winkler oracle':>15}{'oracle gain':>13}")
    for sub, g in d.groupby("subset"):
        print(f"{sub:<8}{g.w_global.mean():>15.2f}{g.w_oracle.mean():>15.2f}{g.gain_pct.mean():>12.2f}%")
    print(f"{'ALL':<8}{d.w_global.mean():>15.2f}{d.w_oracle.mean():>15.2f}{d.gain_pct.mean():>12.2f}%")
    print(f"\nper-fold oracle gain: min {d.gain_pct.min():.2f}%  median {d.gain_pct.median():.2f}%  "
          f"max {d.gain_pct.max():.2f}%")
    print("\nHEADROOM:", "LARGE" if abs(d.gain_pct.mean()) >= 8 else
          ("MODERATE" if abs(d.gain_pct.mean()) >= 3 else "SMALL"),
          "(oracle per-engine allocation vs one global q)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
