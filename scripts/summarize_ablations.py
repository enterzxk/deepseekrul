"""Seed-aware aggregation for the STG-MambaNet final ablation.

The runner stores both the main clip125 metrics and the raw-label appendix:
  test_metrics_clip125  (main table)
  appendix_raw          (appendix only)
Older pilot records only have test_metrics; they are treated as clip125 for
backward compatibility and marked as legacy in the long table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VARIANT_ORDER = (
    "wo_agcn",
    "identity_graph",
    "wo_ssm",
    "wo_physics",
    "wo_mono",
    "full",
    "bilstm",
    "tcn",
    "transformer",
)
VARIANT_LABELS = {
    "wo_agcn": "w/o A-GCN",
    "identity_graph": "Identity-Graph (A=I)",
    "wo_ssm": "w/o SSM-Block (matched LSTM)",
    "wo_physics": "w/o Physics Loss",
    "wo_mono": "w/o Mono Head",
    "full": "Full Model (Ours)",
    "bilstm": "Baseline BiLSTM",
    "tcn": "Baseline TCN",
    "transformer": "Baseline Transformer",
}


def _metric(metrics: dict, key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return default


def collect(run_dirs: list[Path]) -> list[dict]:
    records: dict[tuple[str, int], dict] = {}
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        for metrics_path in sorted(run_dir.glob("seed_*/*/metrics.json")):
            record = json.loads(metrics_path.read_text(encoding="utf-8"))
            key = (str(record["variant"]), int(record["seed"]))
            if key in records:
                old = records[key]
                if old.get("test_metrics") != record.get("test_metrics"):
                    raise ValueError(
                        f"Conflicting duplicate result for variant={key[0]} "
                        f"seed={key[1]}"
                    )
            else:
                records[key] = record
    return [records[key] for key in sorted(records, key=lambda x: (x[1], x[0]))]


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean()), std


def _summary_rows(frame: pd.DataFrame, prefix: str) -> list[dict]:
    rows: list[dict] = []
    for variant in VARIANT_ORDER:
        group = frame.loc[frame["variant"] == variant]
        if group.empty:
            continue
        rmse_mean, rmse_std = _mean_std(group[f"{prefix}rmse"].to_numpy())
        nasa_mean, nasa_std = _mean_std(group[f"{prefix}nasa_score"].to_numpy())
        mae_mean, mae_std = _mean_std(group[f"{prefix}mae"].to_numpy())
        rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "seeds": int(group["seed"].nunique()),
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "nasa_mean": nasa_mean,
                "nasa_std": nasa_std,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "late_fraction_mean": float(group["late_fraction"].mean()),
                "severe_late_count_gt10_mean": float(
                    group["severe_late_count_gt10"].mean()
                ),
                "mono_violation_rate_mean": float(
                    group["mono_violation_rate"].mean()
                ),
                "params": int(group["num_parameters"].iloc[0]),
                "seeds_list": " ".join(
                    str(int(x)) for x in sorted(group["seed"].unique())
                ),
            }
        )
    return rows


def _paired_rows(frame: pd.DataFrame, prefix: str) -> list[dict]:
    rows: list[dict] = []
    if "full" not in set(frame["variant"]):
        return rows
    full = frame.loc[frame["variant"] == "full"].set_index("seed")
    for variant in VARIANT_ORDER:
        if variant == "full":
            continue
        group = frame.loc[frame["variant"] == variant].set_index("seed")
        common = sorted(set(full.index) & set(group.index))
        for seed in common:
            rows.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "seed": int(seed),
                    "delta_rmse_vs_full": float(
                        group.loc[seed, f"{prefix}rmse"]
                        - full.loc[seed, f"{prefix}rmse"]
                    ),
                    "delta_nasa_vs_full": float(
                        group.loc[seed, f"{prefix}nasa_score"]
                        - full.loc[seed, f"{prefix}nasa_score"]
                    ),
                    "delta_late_fraction_vs_full": float(
                        group.loc[seed, "late_fraction"]
                        - full.loc[seed, "late_fraction"]
                    ),
                    "delta_severe_late_count_vs_full": float(
                        group.loc[seed, "severe_late_count_gt10"]
                        - full.loc[seed, "severe_late_count_gt10"]
                    ),
                    "full_rmse": float(full.loc[seed, f"{prefix}rmse"]),
                    "full_nasa": float(full.loc[seed, f"{prefix}nasa_score"]),
                }
            )
    return rows


def _markdown(rows: list[dict], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| Variant | RMSE (mean +/- std) | NASA (mean +/- std) | MAE (mean +/- std) | Late_Fraction | Severe_Late_Count | Mono_Violation_Rate | Params | Seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant_label']} | "
            f"{row['rmse_mean']:.3f} +/- {row['rmse_std']:.3f} | "
            f"{row['nasa_mean']:.3f} +/- {row['nasa_std']:.3f} | "
            f"{row['mae_mean']:.3f} +/- {row['mae_std']:.3f} | "
            f"{row['late_fraction_mean']:.3f} | "
            f"{row['severe_late_count_gt10_mean']:.2f} | "
            f"{row['mono_violation_rate_mean']:.3f} | "
            f"{row['params']:,} | {row['seeds_list']} |"
        )
    return "\n".join(lines) + "\n"


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 10000, seed: int = 2027
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = rng.integers(0, n, size=n)
        means[idx] = float(values[sample].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_statistics(run_dirs: list[Path]) -> list[dict]:
    """Paired per-engine absolute-error tests against the full model."""
    from scipy.stats import wilcoxon

    by_key: dict[tuple[str, int], Path] = {}
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("seed_*/**/test_predictions.csv")):
            variant = path.parent.name
            seed = int(path.parent.parent.name.replace("seed_", ""))
            by_key[(variant, seed)] = path
    rows: list[dict] = []
    full_keys = sorted(seed for (variant, seed) in by_key if variant == "full")
    for variant in VARIANT_ORDER:
        if variant == "full":
            continue
        diffs: list[np.ndarray] = []
        for seed in full_keys:
            full_path = by_key.get(("full", seed))
            variant_path = by_key.get((variant, seed))
            if full_path is None or variant_path is None:
                continue
            full = pd.read_csv(full_path)
            other = pd.read_csv(variant_path)
            if len(full) != len(other):
                continue
            y_true = full["y_true"].to_numpy(dtype=np.float64)
            full_err = np.abs(full["y_pred"].to_numpy(dtype=np.float64) - y_true)
            other_err = np.abs(other["y_pred"].to_numpy(dtype=np.float64) - y_true)
            diffs.append(other_err - full_err)
        if not diffs:
            continue
        diff = np.concatenate(diffs)
        if np.allclose(diff, 0.0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(diff, alternative="two-sided").pvalue)
        low, high = _bootstrap_ci(diff)
        rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "n_pairs": int(diff.size),
                "mean_delta_abs_error": float(diff.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "wilcoxon_p": p_value,
                "significant_0.05": bool(p_value < 0.05),
                "significant_0.01": bool(p_value < 0.01),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = collect(args.run_dirs)
    if not records:
        raise ValueError("No metrics.json files were found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    long_rows = []
    for record in records:
        main = record.get("test_metrics_clip125") or record.get("test_metrics", {})
        raw = record.get("appendix_raw") or record.get("test_metrics", {})
        legacy = "test_metrics_clip125" not in record
        long_rows.append(
            {
                "variant": record["variant"],
                "variant_label": record.get(
                    "variant_label", VARIANT_LABELS.get(record["variant"], record["variant"])
                ),
                "seed": int(record["seed"]),
                "best_epoch": int(record.get("best_epoch", -1)),
                "num_parameters": int(record.get("num_parameters", 0)),
                "legacy_record": legacy,
                "clip_rmse": _metric(main, "rmse"),
                "clip_nasa_score": _metric(main, "nasa_score"),
                "clip_mae": _metric(main, "mae"),
                "raw_rmse": _metric(raw, "rmse"),
                "raw_nasa_score": _metric(raw, "nasa_score"),
                "raw_mae": _metric(raw, "mae"),
                "late_fraction": _metric(main, "late_fraction"),
                "severe_late_count_gt10": _metric(
                    main, "severe_late_count_gt10"
                ),
                "mono_violation_rate": _metric(main, "mono_violation_rate"),
                "mean_bias": _metric(main, "mean_bias"),
            }
        )

    frame = pd.DataFrame(long_rows).sort_values(["variant", "seed"])
    frame.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)

    summary_rows = _summary_rows(frame, "clip_")
    appendix_rows = _summary_rows(frame, "raw_")
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "summary.csv", index=False
    )
    pd.DataFrame(appendix_rows).to_csv(
        args.output_dir / "summary_appendix_raw.csv", index=False
    )
    (args.output_dir / "summary.md").write_text(
        _markdown(summary_rows, "FD001 Final Ablation - clip125 main table"),
        encoding="utf-8",
    )
    (args.output_dir / "summary_appendix_raw.md").write_text(
        _markdown(appendix_rows, "FD001 Final Ablation - raw-label appendix"),
        encoding="utf-8",
    )

    paired_rows = _paired_rows(frame, "clip_")
    raw_paired_rows = _paired_rows(frame, "raw_")
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "paired_deltas_vs_full.csv", index=False
    )
    pd.DataFrame(raw_paired_rows).to_csv(
        args.output_dir / "paired_deltas_vs_full_appendix_raw.csv", index=False
    )
    stats_rows = paired_statistics(args.run_dirs)
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(
            args.output_dir / "paired_stats.csv", index=False
        )

    payload = {
        "run_dirs": [str(path) for path in args.run_dirs],
        "main_protocol": "clip125",
        "appendix_protocol": "raw test RUL",
        "summary": summary_rows,
        "appendix_summary": appendix_rows,
        "paired_deltas": paired_rows,
        "appendix_paired_deltas": raw_paired_rows,
        "paired_statistics": stats_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
