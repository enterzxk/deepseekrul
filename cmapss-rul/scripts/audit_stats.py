"""Per-engine paired bootstrap and honest monotonicity reporting.

Two things this file gets right that the previous version did not:

1. The bootstrap resamples ENGINES and **recomputes RMSE for each model inside
   every resample**, so the interval is for the RMSE difference, not for the
   mean signed error.
2. Single-model-seed-mean and prediction-ensemble are reported separately.

Usage:
    python scripts/audit_stats.py --model-a <glob> --model-b <glob> [--label x]
Globs must match directories containing seed_*/<variant>/test_predictions.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_frames(pattern: str) -> list[pd.DataFrame]:
    return [pd.read_csv(p) for p in sorted(Path(ROOT).glob(pattern + "/test_predictions.csv"))]


def align(frames: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_seeds, n_engines) prediction and (n_engines,) target arrays."""
    units = frames[0]["unit_id"].to_numpy()
    preds, target = [], None
    for frame in frames:
        if not np.array_equal(frame["unit_id"].to_numpy(), units):
            raise ValueError("engine order differs between runs; cannot pair")
        preds.append(frame["y_pred"].to_numpy())
        target = frame["y_true"].to_numpy() if target is None else target
    return np.stack(preds), target


def per_engine_mse(preds: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Seed-averaged per-engine squared error (one value per engine)."""
    return ((preds - target) ** 2).mean(axis=0)


def bootstrap_rmse_difference(
    mse_a: np.ndarray, mse_b: np.ndarray, reps: int = 20000, seed: int = 20261021
) -> dict:
    rng = np.random.default_rng(seed)
    n = mse_a.size
    idx = rng.integers(0, n, size=(reps, n))
    rmse_a = np.sqrt(mse_a[idx].mean(axis=1))
    rmse_b = np.sqrt(mse_b[idx].mean(axis=1))
    diff = rmse_a - rmse_b  # negative => A better
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {
        "n_engines": int(n),
        "rmse_a": float(np.sqrt(mse_a.mean())),
        "rmse_b": float(np.sqrt(mse_b.mean())),
        "rmse_difference": float(np.sqrt(mse_a.mean()) - np.sqrt(mse_b.mean())),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "prob_a_better": float((diff < 0).mean()),
    }


def summarise(pattern: str) -> dict:
    frames = load_frames(pattern)
    if not frames:
        raise SystemExit(f"no runs matched {pattern}")
    preds, target = align(frames)
    seed_rmse = [float(np.sqrt(np.mean((p - target) ** 2))) for p in preds]
    ensemble = preds.mean(axis=0)
    return {
        "pattern": pattern,
        "n_seeds": len(frames),
        "per_seed_rmse": seed_rmse,
        "seed_mean_rmse": float(np.mean(seed_rmse)),
        "seed_std_rmse": float(np.std(seed_rmse)),
        "ensemble_rmse": float(np.sqrt(np.mean((ensemble - target) ** 2))),
        "_mse": per_engine_mse(preds, target),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--out", default="results/audit")
    args = parser.parse_args()

    a = summarise(args.model_a)
    b = summarise(args.model_b)
    boot = bootstrap_rmse_difference(a["_mse"], b["_mse"])
    result = {
        "model_a": {k: v for k, v in a.items() if k != "_mse"},
        "model_b": {k: v for k, v in b.items() if k != "_mse"},
        "paired_engine_bootstrap": boot,
        # No trajectory is stored per window, so within-window monotonicity is
        # undefined for these runs. Online monotonicity needs consecutive
        # windows of the same engine and is computed in Stage 4.
        "mono_violation": "N/A (no per-window trajectory stored)",
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "paired_rmse_bootstrap.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
