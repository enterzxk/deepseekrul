"""Generic subset aggregation for IDA-PE runs (FD003/FD004, e22a_align, ...).

Same statistics as the frozen Gate 0 machinery: pooled engine-level RMSE
(per-unit MSE averaged over units, sqrt) and engine-clustered bootstrap CIs
for contrasts. This is an ANALYSIS script for non-frozen subsets; it does not
modify gates.json or the frozen Gate 0 verdict files.

Usage:
  python summarize_subsets.py --base experiments/ida/gate0 \
      --subsets FD003 FD004 --arms M0 M1 M1p M1_shuf M1_const \
      --contrasts "M1,M0" "M1_shuf,M1" "M1_const,M1" \
      --out experiments/ida/gate0/fd34_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pooled_rmse(df: pd.DataFrame) -> float:
    m = ((df.y_pred - df.y_true) ** 2).groupby(df.unit).mean()
    return float(np.sqrt(m.mean()))


def load_pooled(base: Path, subset: str, arm: str, seed: int) -> pd.DataFrame:
    parts = []
    for fold in range(5):
        d = base / subset / f"{arm}_f{fold}_s{seed}"
        f = d / "val_predictions.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    if not parts:
        raise FileNotFoundError(f"no runs for {subset}/{arm} s{seed} under {base}")
    return pd.concat(parts, ignore_index=True)


def boot_ci_delta(dfa: pd.DataFrame, dfb: pd.DataFrame, reps: int = 5000,
                  seed: int = 20261104) -> tuple:
    """Bootstrap CI for pooled-RMSE(A) - pooled-RMSE(B), resampling engines."""
    ra = dfb.unit.unique()  # common unit space
    units = np.array(sorted(set(dfa.unit.unique()) & set(dfb.unit.unique())))
    rng = np.random.default_rng(seed)
    ma = ((dfa.y_pred - dfa.y_true) ** 2).groupby(dfa.unit).mean()
    mb = ((dfb.y_pred - dfb.y_true) ** 2).groupby(dfb.unit).mean()
    ma, mb = ma.reindex(units), mb.reindex(units)
    idx = rng.integers(0, len(units), size=(reps, len(units)))
    da = np.sqrt(ma.values[idx].mean(axis=1))
    db = np.sqrt(mb.values[idx].mean(axis=1))
    d = da - db
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="experiments/ida/gate0")
    ap.add_argument("--subsets", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--contrasts", nargs="+", default=[],
                    help='pairs "A,B" -> CI for RMSE(A)-RMSE(B)')
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base = Path(args.base)
    result = {}
    for sub in args.subsets:
        arm_rmse = {}
        pool = {}
        for arm in args.arms:
            try:
                df = load_pooled(base, sub, arm, args.seed)
            except FileNotFoundError as e:
                print(f"[skip] {e}")
                continue
            pool[arm] = df
            arm_rmse[arm] = pooled_rmse(df)
        print(f"===== {sub} (seed {args.seed}) =====")
        for arm, r in arm_rmse.items():
            n_units = pool[arm].unit.nunique()
            print(f"  {arm:10s} pooled RMSE {r:7.3f}  (n_units={n_units})")
        contrast = {}
        for c in args.contrasts:
            a, b = c.split(",")
            if a not in pool or b not in pool:
                print(f"  [skip contrast {c}: missing arm]")
                continue
            lo, hi = boot_ci_delta(pool[a], pool[b])
            d = arm_rmse[a] - arm_rmse[b]
            contrast[c] = {"dRMSE": round(d, 4), "ci_low": round(lo, 4),
                           "ci_high": round(hi, 4)}
            sig = "CI-excludes-0" if (lo > 0 or hi < 0) else "CI-crosses-0"
            print(f"  {a} vs {b}: dRMSE {d:+.3f} CI [{lo:+.3f}, {hi:+.3f}]  {sig}")
        result[sub] = {"arm_rmse": {k: round(v, 4) for k, v in arm_rmse.items()},
                       "contrasts": contrast}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
