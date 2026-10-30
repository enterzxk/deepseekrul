"""Zero-retraining multi-seed ensemble and paired Wilcoxon statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

VARIANTS = ("full", "wo_agcn", "identity_graph", "wo_ssm", "wo_physics", "wo_mono")
LABELS = {
    "full": "Full Model",
    "wo_agcn": "w/o A-GCN",
    "identity_graph": "Identity-Graph (A=I)",
    "wo_ssm": "w/o SSM-Block (matched LSTM)",
    "wo_physics": "w/o Physics Loss",
    "wo_mono": "w/o Mono Loss",
}


def per_engine_nasa(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    d = pred - true
    return np.where(d < 0.0, np.expm1(-d / 13.0), np.expm1(d / 10.0))


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a) - np.asarray(b)
    if np.allclose(diff, 0.0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    d = pred - true
    return {
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "nasa_score": float(per_engine_nasa(true, pred).sum()),
        "mae": float(np.mean(np.abs(d))),
        "late_ratio": float(np.mean(d > 0.0)),
        "severe_late_count_gt10": int(np.sum(d > 10.0)),
        "severe_late_ratio_gt10": float(np.mean(d > 10.0)),
        "n_engines": int(len(d)),
    }


def load_frames(root: Path, subset: str, variant: str) -> list[pd.DataFrame]:
    pattern = f"seed_*/seed_*/{variant}/test_predictions.csv" if subset == "FD001" else f"seed_*/seed_*/{variant}/test_predictions.csv"
    files = sorted(root.glob(pattern))
    frames = []
    for path in files:
        frame = pd.read_csv(path).sort_values("unit_id").reset_index(drop=True)
        frame["_file"] = str(path)
        frames.append(frame)
    return frames


def analyze_subset(root: Path, subset: str, output_dir: Path) -> list[dict]:
    rows = []
    ensemble_predictions = {}
    per_seed_abs = {}
    per_seed_nasa = {}
    true_reference = None
    for variant in VARIANTS:
        frames = load_frames(root, subset, variant)
        if not frames:
            continue
        n_expected = 5 if subset == "FD001" else 3
        if len(frames) != n_expected:
            print(f"WARN {subset}/{variant}: found {len(frames)} seeds, expected {n_expected}")
        unit_ids = frames[0]["unit_id"].to_numpy()
        true = frames[0]["y_true"].to_numpy(dtype=float)
        for frame in frames[1:]:
            if not np.array_equal(frame["unit_id"].to_numpy(), unit_ids):
                raise ValueError(f"Unit mismatch in {subset}/{variant}")
            if not np.allclose(frame["y_true"].to_numpy(dtype=float), true):
                raise ValueError(f"Target mismatch in {subset}/{variant}")
        predictions = np.stack(
            [frame["y_pred"].to_numpy(dtype=float) for frame in frames], axis=0
        )
        predictions = np.clip(predictions, 0.0, 125.0)
        ensemble = predictions.mean(axis=0)
        ensemble_predictions[variant] = (unit_ids, true, ensemble)
        per_seed_abs[variant] = np.mean(np.abs(predictions - true[None, :]), axis=0)
        per_seed_nasa[variant] = np.mean(
            np.stack([per_engine_nasa(true, row) for row in predictions], axis=0), axis=0
        )
        if true_reference is None:
            true_reference = true
        m = metrics(true, ensemble)
        rows.append(
            {
                "subset": subset,
                "variant": variant,
                "variant_label": LABELS[variant],
                "evaluation": "ensemble",
                "n_seeds": len(frames),
                **m,
                "p_abs_error_vs_full": np.nan,
                "p_nasa_vs_full": np.nan,
            }
        )
        pd.DataFrame(
            {
                "unit_id": unit_ids,
                "y_true": true,
                "y_pred": ensemble,
                "signed_error": ensemble - true,
            }
        ).to_csv(output_dir / f"{subset}_{variant}_ensemble.csv", index=False)

    if true_reference is None or "full" not in ensemble_predictions:
        return rows
    full_unit, full_true, full_ens = ensemble_predictions["full"]
    full_abs_ens = np.abs(full_ens - full_true)
    full_nasa_ens = per_engine_nasa(full_true, full_ens)
    full_abs_single = per_seed_abs["full"]
    full_nasa_single = per_seed_nasa["full"]
    for row in rows:
        variant = row["variant"]
        abs_ens = np.abs(ensemble_predictions[variant][2] - full_true)
        nasa_ens = per_engine_nasa(full_true, ensemble_predictions[variant][2])
        row["p_abs_error_vs_full"] = safe_wilcoxon(full_abs_ens, abs_ens)
        row["p_nasa_vs_full"] = safe_wilcoxon(full_nasa_ens, nasa_ens)
        row["p_abs_error_single_vs_full"] = safe_wilcoxon(
            full_abs_single, per_seed_abs[variant]
        )
        row["p_nasa_single_vs_full"] = safe_wilcoxon(
            full_nasa_single, per_seed_nasa[variant]
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root
    output = args.output or root / "results" / "final_ensemble_and_pvalues.csv"
    ensemble_dir = root / "results" / "ensemble_predictions"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(analyze_subset(root / "experiments" / "final" / "FD001_locked", "FD001", ensemble_dir))
    rows.extend(analyze_subset(root / "experiments" / "final" / "FD002_3seeds", "FD002", ensemble_dir))
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
