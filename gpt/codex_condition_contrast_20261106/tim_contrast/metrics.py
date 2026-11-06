"""Exact four-metric evaluation, with engines rather than windows weighted equally."""
from __future__ import annotations

import numpy as np
import pandas as pd


def engine_statistics(predictions):
    required = {"unit", "end_cycle", "y_true", "y_pred"}
    if not required.issubset(predictions):
        raise ValueError(f"Missing columns: {required-set(predictions)}")
    if predictions.duplicated(["unit", "end_cycle"]).any():
        raise ValueError("Duplicate engine/query keys")
    p = predictions.copy()
    error = p.y_pred.to_numpy(np.float64) - p.y_true.to_numpy(np.float64)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Empty or non-finite predictions: never silently drop failed queries")
    with np.errstate(over="raise", invalid="raise"):
        nasa = np.expm1(np.where(error > 0, error/10., -error/13.))
    p["mse"], p["mae"], p["nasa"], p["bias"] = error**2, np.abs(error), nasa, error
    p["late20"] = error > 20.0
    out = p.groupby("unit", sort=True).agg(mse=("mse", "mean"), mae=("mae", "mean"),
           nasa=("nasa", "mean"), bias=("bias", "mean"), late20=("late20", "mean"),
           max_overestimate=("bias", "max"), queries=("end_cycle", "size"))
    return out


def summarize(stats):
    return {"rmse": float(np.sqrt(stats.mse.mean())), "mae": float(stats.mae.mean()),
            "nasa": float(stats.nasa.mean()), "bias": float(stats.bias.mean()),
            "late20": float(stats.late20.mean()),
            "max_overestimate": float(stats.max_overestimate.max()),
            "top3_engine_nasa_share": float(stats.nasa.nlargest(3).sum()/max(stats.nasa.sum(), 1e-300)),
            "engines": len(stats), "queries": int(stats.queries.sum())}


def paired_comparison(candidate, reference, reps=5000, seed=20261106):
    """Pair engine contributions, not windows. Difference is candidate-reference."""
    if not candidate.index.equals(reference.index):
        raise ValueError("Engine sets/order differ")
    rng = np.random.default_rng(seed)
    a, b = candidate.mse.to_numpy(), reference.mse.to_numpy()
    index = rng.integers(0, len(a), (reps, len(a)))
    diff = np.sqrt(a[index].mean(1)) - np.sqrt(b[index].mean(1))
    ca, re = summarize(candidate), summarize(reference)
    return {"gain_percent": 100*(1-ca["rmse"]/re["rmse"]),
            "delta_rmse_cycles": ca["rmse"]-re["rmse"],
            "delta_rmse_ci95_cycles": np.quantile(diff, [.025, .975]).tolist(),
            "mae_ratio": ca["mae"]/re["mae"], "nasa_ratio": ca["nasa"]/re["nasa"],
            "delta_abs_bias_cycles": abs(ca["bias"])-abs(re["bias"])}
