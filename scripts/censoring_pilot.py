"""Pilot: does right-censoring of the CALIBRATION set break standard conformal RUL?

Motivation (docs/GAP_ANALYSIS.md, gap G1): in the field, many engines are still
running when the observation window closes, so their failure time is censored.
Standard conformal RUL calibration needs exact RUL labels; the usual quick fix is
to drop censored engines.  Dropping them is not random: engines still alive at the
horizon are the long-lived ones, so the calibration set becomes biased towards
short lifetimes.  This script measures the resulting coverage/width bias on the
frozen predictions, with no training and no GPU.

Simulation (per subset x fold, seed-42 cell):
  * calibration/test engine split exactly as in the frozen protocol (seed 42, 30%)
  * engine lifetime = max(cutoff) of that engine (full trajectories in the folds)
  * censor at the tau-quantile of CALIBRATION lifetimes: engines with
    lifetime > c* are treated as censored and dropped
  * calibrate the CQR quantile on the surviving calibration windows
  * evaluate coverage / Winkler on ALL test windows (the deployment population)

If coverage drops materially as tau decreases, G1 has a real problem to solve.

Run: python scripts/censoring_pilot.py [--cells-seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
TAUS = [1.0, 0.75, 0.50, 0.25]   # 1.0 = no censoring


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, a=0.1):
    return float(np.mean((hi - lo) + (2.0 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def cqr_quantile(y, lo, hi, mask, delta=0.1):
    s = np.maximum(lo[mask] - y[mask], y[mask] - hi[mask])
    return float(np.quantile(s, min(1.0, (1 - delta) * (1 + 1 / s.size)), method="higher"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for sub in SUBSETS:
        for fold in range(5):
            p = root / "experiments/punet_matrix" / sub / f"f{fold}_s{args.seed}" / "preds_cqr.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            units = df.unit.to_numpy()
            cal, tst = engine_split(units)
            life = df.groupby("unit")["cutoff"].max()
            cal_units = np.unique(units[cal])
            cal_life = life.loc[cal_units].to_numpy()

            for tau in TAUS:
                if tau >= 1.0:
                    keep = np.ones(cal_units.size, bool)
                    c_star = np.inf
                else:
                    c_star = float(np.quantile(cal_life, tau))
                    keep = cal_life <= c_star          # censored engines are dropped

                # ---- arm 1: naive drop of censored engines -------------------
                kept_units = set(cal_units[keep].tolist())
                cal_mask = np.isin(units, list(kept_units))
                if cal_mask.sum() < 50:
                    continue
                q = cqr_quantile(df.y_true.to_numpy(), df.y_lo.to_numpy(), df.y_hi.to_numpy(), cal_mask)

                # ---- arm 2: clip-aware -- keep the CAPPED windows of censored engines
                # If the observation ends at c* and the label is clipped at 125, then for
                # any window with cutoff t <= c* - 125 the true RUL is >= 125, so the
                # (clipped) label is exactly 125 and is NOT censored.  Dropping the whole
                # engine throws these known-label, low-score windows away.
                cap = 125.0
                censored_units = set(cal_units[~keep].tolist())
                clip_mask = np.isin(units, list(kept_units))
                if np.isfinite(c_star) and censored_units:
                    in_cens = np.isin(units, list(censored_units))
                    known_capped = in_cens & (df.cutoff.to_numpy() <= c_star - cap) & (df.y_true.to_numpy() >= cap - 1e-6)
                    clip_mask = clip_mask | known_capped
                q_clip = cqr_quantile(df.y_true.to_numpy(), df.y_lo.to_numpy(), df.y_hi.to_numpy(), clip_mask) \
                    if clip_mask.sum() >= 50 else np.nan

                lo = df.y_lo.to_numpy()[tst] - q
                hi = df.y_hi.to_numpy()[tst] + q
                y = df.y_true.to_numpy()[tst]
                lo_c = df.y_lo.to_numpy()[tst] - q_clip
                hi_c = df.y_hi.to_numpy()[tst] + q_clip
                rows.append(dict(subset=sub, fold=fold, tau=tau, q=q, q_clip=q_clip,
                                 picp=float(np.mean((y >= lo) & (y <= hi))),
                                 winkler=winkler(y, lo, hi),
                                 picp_clip=float(np.mean((y >= lo_c) & (y <= hi_c))),
                                 winkler_clip=winkler(y, lo_c, hi_c),
                                 n_cal=int(cal_mask.sum()), n_clip=int(clip_mask.sum()),
                                 n_units=len(kept_units)))

    d = pd.DataFrame(rows)
    if d.empty:
        print("no cells found")
        return 1
    base = d[d.tau == 1.0].set_index(["subset", "fold"])
    print(f"{'tau':>5} {'calib eng':>9} {'PICP':>7} {'dPICP':>7} {'Winkler':>8} {'dWink%':>7} | "
          f"{'PICP_clip':>9} {'Wink_clip':>9} {'dWink_clip%':>11}")
    summary = []
    for tau in TAUS:
        g = d[d.tau == tau].set_index(["subset", "fold"])
        common = g.index.intersection(base.index)
        if len(common) == 0:
            continue
        picp = g.loc[common, "picp"].mean()
        w = g.loc[common, "winkler"].mean()
        dp = picp - base.loc[common, "picp"].mean()
        dw = (w - base.loc[common, "winkler"].mean()) / base.loc[common, "winkler"].mean() * 100
        pc = g.loc[common, "picp_clip"].mean()
        wc = g.loc[common, "winkler_clip"].mean()
        dwc = (wc - base.loc[common, "winkler"].mean()) / base.loc[common, "winkler"].mean() * 100
        nu = g.loc[common, "n_units"].mean()
        summary.append((tau, nu, picp, dp, w, dw, pc, wc, dwc))
        print(f"{tau:>5.2f} {nu:>9.1f} {picp:>7.4f} {dp:>+7.4f} {w:>8.2f} {dw:>+7.2f} | "
              f"{pc:>9.4f} {wc:>9.2f} {dwc:>+11.2f}")

    # the pilot is informative if censoring at tau=0.5 moves coverage by >= 1 pp
    mid = [s for s in summary if abs(s[0] - 0.5) < 1e-9]
    if mid:
        tau, nu, picp, dp, w, dw, pc, wc, dwc = mid[0]
        print(f"\nAt tau=0.50:")
        print(f"  naive drop : coverage {picp:.4f} ({dp:+.4f}), Winkler {dw:+.2f}%")
        print(f"  clip-aware : coverage {pc:.4f} ({pc - base['picp'].mean():+.4f}), Winkler {dwc:+.2f}%")
        print(f"  calibration engines: {nu:.1f}")
        print("G1 PREMISE:", "SUPPORTED" if abs(dp) >= 0.01 else "WEAK",
              "(|dPICP| >= 1 pp means dropping censored engines materially breaks calibration)")
        print("EFFICIENCY RECOVERY:", "YES" if dwc <= 0.5 * dw else "NO",
              "(clip-aware must cut the Winkler loss by at least half)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
