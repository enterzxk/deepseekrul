"""Does including each engine's final window (RUL = 0) inflate per-engine coverage?

Parallel-line D032 claims:
  - `preds*.csv` ends at the engine's life end (max_cut == T), where RUL = 0;
  - using that window inflates realised coverage (frozen PICP looks ~0.9);
  - with test-like truncation (window whose y_true is closest to 60) PICP drops to ~0.80.

This script recomputes per-engine coverage three ways on the frozen CQR predictions:
  A. all windows (what the current analysis does)
  B. drop each engine's last window (the RUL = 0 endpoint)
  C. test-like truncation: keep only the window whose y_true is closest to 60

If A > B materially, the published per-engine coverage numbers are optimistic and the
measurement-protocol issue is real. Run on a40 (predictions live there).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = "experiments/punet_matrix"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 7, 2024, 13]
DELTA = 0.1
SPLIT_SEED = 42


def split(units, frac=0.3, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def main() -> int:
    rows = []
    for sub in SUBSETS:
        for seed in SEEDS:
            for fold in range(5):
                d = os.path.join(ROOT, sub, f"f{fold}_s{seed}")
                p = os.path.join(d, "preds_cqr.csv")
                if not os.path.exists(p):
                    continue
                df = pd.read_csv(p)
                cal, tst = split(df["unit"].to_numpy())
                s = np.maximum(df["y_lo"].to_numpy()[cal] - df["y_true"].to_numpy()[cal],
                               df["y_true"].to_numpy()[cal] - df["y_hi"].to_numpy()[cal])
                lv = min(1.0, (1 - DELTA) * (1 + 1 / s.size))
                q = float(np.quantile(s, lv, method="higher"))
                hit = ((df["y_true"] >= df["y_lo"] - q) & (df["y_true"] <= df["y_hi"] + q)).to_numpy()
                d2 = df.assign(hit=hit)[tst]

                def engine_cov(frame):
                    g = frame.groupby("unit")["hit"].mean()
                    return float(g.mean()), float(g.min()), len(g)

                # A: all windows
                A = engine_cov(d2)
                # B: drop each engine's last window (highest cutoff)
                last = d2.groupby("unit")["cutoff"].idxmax()
                B = engine_cov(d2.drop(index=last))
                # C: test-like truncation, one window per engine closest to y_true = 60
                idx = (d2["y_true"] - 60).abs().groupby(d2["unit"]).idxmin()
                C = engine_cov(d2.loc[idx])
                rows.append(dict(subset=sub, seed=seed, fold=fold,
                                 picp_all=A[0], picp_droplast=B[0], picp_tl60=C[0],
                                 min_all=A[1], min_droplast=B[1], min_tl60=C[1],
                                 n_eng=A[2],
                                 n_win_all=len(d2), n_win_tl=len(C and idx)))
    out = pd.DataFrame(rows)
    out.to_csv("experiments/truncation_check.csv", index=False)
    print(f"cells: {len(out)}")
    print(out.groupby("subset")[["picp_all", "picp_droplast", "picp_tl60",
                                 "min_all", "min_droplast", "min_tl60"]].mean().round(3).to_string())
    print("\nwindows per cell: all %.0f | test-like %.0f" % (out.n_win_all.mean(), out.n_win_tl.mean()))
    print("\nmean over cells: all %.3f | drop-last %.3f | test-like %.3f"
          % (out.picp_all.mean(), out.picp_droplast.mean(), out.picp_tl60.mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
