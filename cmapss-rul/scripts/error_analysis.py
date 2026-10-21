"""Per-engine and per-RUL-band error analysis (ORDER_01 Task B).

Reads the deployment-style pseudo-test predictions written by
scripts/run_candidate.py (one row per engine per cutoff) and reports the error
structure that mean RMSE hides: which lifetime bands and which engines carry
the loss, and whether the ranking against the Transformer control flips by band.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BANDS = [(0, 25), (25, 50), (50, 100), (100, 126)]


def nasa(delta: np.ndarray) -> float:
    return float(np.sum(np.where(delta < 0, np.expm1(-delta / 13.0), np.expm1(delta / 10.0))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/order01")
    parser.add_argument("--out", default="results/order01")
    args = parser.parse_args()
    root = ROOT / args.root
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(root.glob("*/*/pseudo_test_predictions.csv")):
        subset, run = path.parts[-3], path.parts[-2]
        variant, seed = run.rsplit("_s", 1)
        frame = pd.read_csv(path)
        frame["variant"], frame["seed"], frame["subset"] = variant, int(seed), subset
        rows.append(frame)
    if not rows:
        raise SystemExit("no pseudo_test_predictions.csv found")
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(out / "pseudo_predictions_all.csv", index=False)

    # ---- aggregate per (subset, variant) at two views: per-cutoff and per-engine
    records = []
    for (subset, variant), group in data.groupby(["subset", "variant"]):
        err = group["signed_error"].to_numpy()
        records.append({
            "subset": subset, "variant": variant, "n_pred": len(group),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "nasa": nasa(err), "bias": float(err.mean()),
            "late_fraction": float((err > 0).mean()),
            "severe_late_gt10": int((err > 10).sum()),
        })
    summary = pd.DataFrame(records).sort_values(["subset", "rmse"])
    summary.to_csv(out / "pseudo_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- band analysis for the Track A / tf comparison
    band_rows = []
    for (subset, variant), group in data.groupby(["subset", "variant"]):
        for lo, hi in BANDS:
            part = group[(group.y_true >= lo) & (group.y_true < hi)]
            if part.empty:
                continue
            err = part["signed_error"].to_numpy()
            band_rows.append({
                "subset": subset, "variant": variant, "band": f"[{lo},{hi})",
                "n": len(part), "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))), "bias": float(err.mean()),
            })
    bands = pd.DataFrame(band_rows)
    bands.to_csv(out / "band_errors.csv", index=False)

    # ---- worst engines
    per_engine = (
        data.assign(sq=data.signed_error ** 2)
        .groupby(["subset", "variant", "unit_id"])
        .agg(rmse=("sq", lambda s: float(np.sqrt(s.mean()))),
             bias=("signed_error", "mean"),
             n=("sq", "size"))
        .reset_index()
    )
    per_engine.sort_values(["subset", "rmse"], ascending=[True, False]).to_csv(
        out / "per_engine_rmse.csv", index=False)
    print("\nband table:")
    print(bands.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
