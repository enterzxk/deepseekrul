"""Engine-clustered paired bootstrap (ORDER_02 Issue 5).

Multiple cutoffs and multiple seeds per engine are NOT independent samples.
The bootstrap resamples ENGINES and keeps every cutoff/seed of a sampled engine
together, recomputing RMSE for both models inside each resample.

Usage:
  python scripts/audit_stats.py --a "experiments/order01/FD002/Aid_s*/pseudo_test_predictions.csv" \
                                --b "experiments/order01/FD002/tf_s*/pseudo_test_predictions.csv"
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / pattern)))]
    if not frames:
        raise SystemExit(f"no files matched {pattern}")
    return pd.concat(frames, ignore_index=True)


def per_engine_sse(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("unit_id").apply(
        lambda g: float((g.signed_error ** 2).sum()), include_groups=False
    )


def per_engine_count(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("unit_id").size()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", default="results/order02")
    parser.add_argument("--reps", type=int, default=20000)
    args = parser.parse_args()

    fa, fb = load(args.a), load(args.b)
    sse_a, cnt_a = per_engine_sse(fa), per_engine_count(fa)
    sse_b, cnt_b = per_engine_sse(fb), per_engine_count(fb)
    units = sorted(set(sse_a.index) & set(sse_b.index))
    sa, sb = sse_a.loc[units].to_numpy(), sse_b.loc[units].to_numpy()
    ca, cb = cnt_a.loc[units].to_numpy(), cnt_b.loc[units].to_numpy()

    def rmse(sse, cnt, idx):
        return np.sqrt(sse[idx].sum() / cnt[idx].sum())

    rng = np.random.default_rng(20261021)
    idx = rng.integers(0, len(units), size=(args.reps, len(units)))
    diffs = np.array([rmse(sa, ca, i) - rmse(sb, cb, i) for i in idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    result = {
        "a": args.a, "b": args.b,
        "n_engines": len(units),
        "n_predictions_a": int(ca.sum()), "n_predictions_b": int(cb.sum()),
        "rmse_a": float(rmse(sa, ca, np.arange(len(units)))),
        "rmse_b": float(rmse(sb, cb, np.arange(len(units)))),
        "rmse_diff": float(rmse(sa, ca, np.arange(len(units))) - rmse(sb, cb, np.arange(len(units)))),
        "ci_low": float(lo), "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "prob_a_better": float((diffs < 0).mean()),
        "clustering": "engines resampled; all cutoffs/seeds of an engine stay together",
    }
    # seed-level spread, reported separately
    if "seed" in fa.columns:
        result["per_seed_rmse_a"] = {
            str(s): float(np.sqrt((g.signed_error ** 2).mean())) for s, g in fa.groupby("seed")
        }
        result["per_seed_rmse_b"] = {
            str(s): float(np.sqrt((g.signed_error ** 2).mean())) for s, g in fb.groupby("seed")
        }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "paired_bootstrap.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
