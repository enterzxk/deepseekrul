"""Paired Wilcoxon tests from per-engine prediction CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def per_engine_metrics(frame: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    rows = []
    for unit_id, group in frame.groupby("unit_id"):
        true = group["y_true"].to_numpy(dtype=float)
        pred = group[prediction_col].to_numpy(dtype=float)
        delta = pred - true
        nasa = np.where(delta < 0, np.expm1(-delta / 13), np.expm1(delta / 10)).sum()
        rows.append(
            {
                "unit_id": int(unit_id),
                "abs_error": float(np.mean(np.abs(delta))),
                "nasa": float(nasa),
                "rmse": float(np.sqrt(np.mean(delta ** 2))),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-col", default="y_ensemble")
    parser.add_argument("--compare", action="append", required=True, nargs=2, metavar=("NAME", "CSV"))
    parser.add_argument("--compare-col", default="y_ensemble")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    ref_metrics = per_engine_metrics(reference, args.reference_col)
    rows = []
    for name, csv_path in args.compare:
        frame = pd.read_csv(csv_path)
        other = per_engine_metrics(frame, args.compare_col)
        merged = ref_metrics.merge(other, on="unit_id", suffixes=("_ref", "_cmp"))
        for metric in ("abs_error", "nasa", "rmse"):
            ref_values = merged[f"{metric}_ref"].to_numpy()
            cmp_values = merged[f"{metric}_cmp"].to_numpy()
            differences = ref_values - cmp_values
            if np.allclose(differences, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(ref_values, cmp_values, zero_method="wilcox", alternative="two-sided")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "reference_mean": float(ref_values.mean()),
                    "comparison_mean": float(cmp_values.mean()),
                    "mean_difference_ref_minus_cmp": float(differences.mean()),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "n_engines": int(len(merged)),
                    "reference_better_engines": int(np.sum(differences < 0)),
                    "comparison_better_engines": int(np.sum(differences > 0)),
                }
            )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
