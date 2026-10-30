"""Aggregate validation-only tuning runs (no test data involved)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/seed_*/validation_metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        metrics = record["best_metrics"]
        rows.append(
            {
                "tag": record["tag"],
                "seed": int(record["seed"]),
                "best_epoch": int(record["best_epoch"]),
                "epochs_ran": int(record["epochs_ran"]),
                "params": int(record["num_parameters"]),
                "val_rmse": float(metrics["val_rmse"]),
                "val_nasa": float(metrics["val_nasa"]),
                "val_mae": float(metrics["val_mae"]),
                "val_late_fraction": float(metrics["val_late_fraction"]),
                "val_severe_late_count_gt10": int(
                    metrics["val_severe_late_count_gt10"]
                ),
                "val_mono_violation_rate": float(
                    metrics["val_mono_violation_rate"]
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/validation_tuning"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.root / "summary")
    output.mkdir(parents=True, exist_ok=True)

    rows = collect(args.root)
    if not rows:
        raise ValueError(f"No validation_metrics.json under {args.root}")
    frame = pd.DataFrame(rows).sort_values(["tag", "seed"])
    frame.to_csv(output / "per_seed.csv", index=False)

    summary = []
    for tag, group in frame.groupby("tag", sort=True):
        def ms(column: str) -> tuple[float, float]:
            values = group[column].to_numpy(dtype=np.float64)
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            return float(values.mean()), std

        rmse_mean, rmse_std = ms("val_rmse")
        nasa_mean, nasa_std = ms("val_nasa")
        summary.append(
            {
                "tag": tag,
                "seeds": int(group["seed"].nunique()),
                "val_rmse_mean": rmse_mean,
                "val_rmse_std": rmse_std,
                "val_nasa_mean": nasa_mean,
                "val_nasa_std": nasa_std,
                "val_mae_mean": float(group["val_mae"].mean()),
                "val_late_fraction_mean": float(group["val_late_fraction"].mean()),
                "val_severe_late_count_mean": float(
                    group["val_severe_late_count_gt10"].mean()
                ),
                "val_mono_violation_mean": float(
                    group["val_mono_violation_rate"].mean()
                ),
                "params": int(group["params"].iloc[0]),
                "best_epoch_mean": float(group["best_epoch"].mean()),
            }
        )
    summary_frame = pd.DataFrame(summary).sort_values("val_rmse_mean")
    summary_frame.to_csv(output / "summary.csv", index=False)

    lines = [
        "| Tag | val RMSE (mean +/- std) | val NASA (mean +/- std) | val MAE | Late_Frac | Severe_Late_Count | Mono_Violation | Params | Seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_frame.to_dict("records"):
        lines.append(
            f"| {row['tag']} | {row['val_rmse_mean']:.3f} +/- {row['val_rmse_std']:.3f} | "
            f"{row['val_nasa_mean']:.2f} +/- {row['val_nasa_std']:.2f} | "
            f"{row['val_mae_mean']:.3f} | {row['val_late_fraction_mean']:.3f} | "
            f"{row['val_severe_late_count_mean']:.2f} | "
            f"{row['val_mono_violation_mean']:.3f} | "
            f"{row['params']:,} | {row['seeds']} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps({"rows": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print((output / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
