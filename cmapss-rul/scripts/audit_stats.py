"""Per-engine paired bootstrap and honest monotonicity reporting.

Seed-level Wilcoxon with n=5 cannot reach p<0.05, so significance must be
established at the engine level: each test engine contributes one prediction,
and the paired difference between two models is bootstrapped over engines.

Usage:
    python scripts/audit_stats.py --models dirA dirB [--label "A vs B"]
Each directory must contain seed_*/<variant>/test_predictions.csv produced by
scripts/run_ablation.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_runs(pattern: str) -> list[pd.DataFrame]:
    frames = []
    for path in sorted(Path(ROOT).glob(pattern)):
        frames.append(pd.read_csv(path))
    return frames


def per_engine_errors(frames: list[pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
    """Average per-engine signed error across seeds (engines are aligned)."""
    unit = frames[0]["unit_id"].to_numpy()
    errors = np.stack([(f["y_pred"] - f["y_true"]).to_numpy() for f in frames])
    return errors.mean(axis=0), unit


def paired_bootstrap(a: np.ndarray, b: np.ndarray, reps: int = 20000) -> dict:
    diff = a - b
    rng = np.random.default_rng(20261021)
    idx = rng.integers(0, diff.size, size=(reps, diff.size))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_engines": int(diff.size),
        "mean_delta": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True, help="glob, e.g. experiments/p10/final2/FD001/p10_full_s*/seed_*/*")
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--out", default="results/audit")
    args = parser.parse_args()

    fa = load_runs(args.model_a + "/test_predictions.csv")
    fb = load_runs(args.model_b + "/test_predictions.csv")
    if not fa or not fb:
        raise SystemExit("no test_predictions.csv matched; check the globs")

    ea, units = per_engine_errors(fa)
    eb, _ = per_engine_errors(fb)
    result = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "seeds_a": len(fa),
        "seeds_b": len(fb),
        "rmse_a": float(np.sqrt(np.mean(ea ** 2))),
        "rmse_b": float(np.sqrt(np.mean(eb ** 2))),
        "rmse_seed_std_a": float(np.std([np.sqrt(np.mean((f["y_pred"] - f["y_true"]) ** 2)) for f in fa])),
        "rmse_seed_std_b": float(np.std([np.sqrt(np.mean((f["y_pred"] - f["y_true"]) ** 2)) for f in fb])),
        "paired_bootstrap": paired_bootstrap(ea, eb),
        # no trajectory is stored per engine, so within-window monotonicity is
        # not defined for these runs; online monotonicity needs consecutive
        # windows from the same engine and is computed in Stage 4.
        "mono_violation": "N/A (no per-window trajectory stored)",
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "paired_bootstrap.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
