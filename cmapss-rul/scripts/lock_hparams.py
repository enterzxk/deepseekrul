"""Freeze P1 architecture or P2 loss weights from validation-only summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments" / "validation_tuning" / "summary" / "summary.csv"
OUT = ROOT / "results" / "locked_hparams.json"


def _config_for_tag(tag: str) -> dict:
    root = ROOT / "experiments" / "validation_tuning" / tag
    for path in sorted(root.glob("seed_*/validation_metrics.json")):
        return json.loads(path.read_text(encoding="utf-8"))["config"]
    raise FileNotFoundError(f"No validation_metrics.json for {tag}")


def p1() -> dict:
    frame = pd.read_csv(SUMMARY)
    frame = frame[frame["tag"].str.startswith("p1")].copy()
    if frame.empty:
        raise ValueError("No P1 rows in summary")
    frame["selection_score"] = frame["val_rmse_mean"] + 0.25 * frame["val_rmse_std"]
    row = frame.sort_values(["selection_score", "val_rmse_mean"]).iloc[0]
    cfg = _config_for_tag(str(row["tag"]))
    return {
        "phase": "P1",
        "selected_tag": str(row["tag"]),
        "window_size": int(cfg["window_size"]),
        "dropout": float(cfg["dropout"]),
        "weight_decay": float(cfg["weight_decay"]),
        "d_model": int(cfg["d_model"]),
        "batch_size": int(cfg["batch_size"]),
        "lr": float(cfg["lr"]),
        "epochs": int(cfg["epochs"]),
        "patience": int(cfg["patience"]),
        "seeds": [42, 67, 123],
        "val_rmse_mean": float(row["val_rmse_mean"]),
        "val_rmse_std": float(row["val_rmse_std"]),
        "val_nasa_mean": float(row["val_nasa_mean"]),
        "selection_rule": "min(val_rmse_mean + 0.25*val_rmse_std)",
    }


def p2() -> dict:
    frame = pd.read_csv(SUMMARY)
    frame = frame[frame["tag"].str.startswith("p2_")].copy()
    full = frame[frame["tag"].str.startswith("p2_full_")].sort_values(
        ["val_rmse_mean", "val_nasa_mean"]
    )
    asym = frame[frame["tag"].str.startswith("p2_asym_")].sort_values(
        ["val_rmse_mean", "val_nasa_mean"]
    )
    if full.empty or asym.empty:
        raise ValueError("P2 full/asym rows missing")
    full_row = full.iloc[0]
    full_cfg = _config_for_tag(str(full_row["tag"]))
    asym_row = asym.iloc[0]
    asym_cfg = _config_for_tag(str(asym_row["tag"]))
    return {
        "phase": "P2",
        "full_loss_tag": str(full_row["tag"]),
        "full_loss": {
            "mode": "full",
            "asym_weight": float(full_cfg["asymmetric_weight"]),
            "lambda_mono": float(full_cfg["lambda_mono"]),
            "late_weight": float(full_cfg["late_weight"]),
            "val_rmse_mean": float(full_row["val_rmse_mean"]),
            "val_rmse_std": float(full_row["val_rmse_std"]),
            "val_nasa_mean": float(full_row["val_nasa_mean"]),
        },
        "wo_mono_loss_tag": str(asym_row["tag"]),
        "wo_mono_loss": {
            "mode": "asym",
            "asym_weight": float(asym_cfg["asymmetric_weight"]),
            "late_weight": float(asym_cfg["late_weight"]),
            "val_rmse_mean": float(asym_row["val_rmse_mean"]),
            "val_rmse_std": float(asym_row["val_rmse_std"]),
            "val_nasa_mean": float(asym_row["val_nasa_mean"]),
        },
        "wo_physics_loss": {"mode": "mse"},
        "selection_rule": "best full-physics and asym tags by validation RMSE, then NASA",
    }


def main() -> int:
    global SUMMARY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("p1", "p2"), required=True)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    SUMMARY = args.summary
    payload = p1() if args.phase == "p1" else p2()
    if args.phase == "p2" and args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        existing.update(payload)
        payload = existing
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
