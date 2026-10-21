"""Build FINAL_PAPER_REPORT.md from actual Phase II artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phase = args.phase_dir

    stage2 = read_json(phase / "stage2_bidirectional_5fold" / "summary.json")
    pareto_path = phase / "pareto_tradeoff.csv"
    pareto = pd.read_csv(pareto_path) if pareto_path.is_file() else None
    final_metrics = read_json(phase / "final" / "FD001" / "metrics.json")
    freeze_manifest = phase / "test_freeze_manifest.sha256"

    lines = [
        "# FINAL PAPER REPORT — Phase II",
        "",
        "Generated from stored artifacts only. Missing stages are marked MISSING; "
        "failed gates are not rewritten as successes.",
        "",
        "## Gate status",
        "",
    ]

    gate1 = stage2 is not None and not stage2.get("test_files_accessed", True)
    gate2 = stage2 is not None and stage2.get("rmse_mean", 999.0) <= 12.8
    gate3 = False
    if pareto is not None and "nasa_score" in pareto and len(pareto) > 0:
        # Gate 3 requires an explicit state-aware versus MSE comparison; the
        # Pareto table alone is not sufficient unless a gamma=1, eta=0 row is present.
        has_mse = bool(((pareto["gamma"] == 1.0) & (pareto["eta"] == 0.0)).any())
        best_nasa = float(pareto["nasa_score"].min())
        baseline_nasa = pareto.loc[
            (pareto["gamma"] == 1.0) & (pareto["eta"] == 0.0), "nasa_score"
        ]
        if has_mse and len(baseline_nasa):
            gate3 = best_nasa <= 0.8 * float(baseline_nasa.iloc[0])
    gate5 = final_metrics is not None

    lines.extend(
        [
            f"- Blind test freeze: {'PASS' if freeze_manifest.is_file() else 'MISSING'}",
            f"- Gate 1 (5-fold, no test access): {'PASS' if gate1 else 'FAIL/MISSING'}",
            f"- Gate 2 (Val RMSE <= 12.8): {'PASS' if gate2 else 'FAIL/MISSING'}",
            f"- Gate 3 (NASA reduction >20%): {'PASS' if gate3 else 'FAIL/MISSING'}",
            f"- Stage 5 blind evaluation: {'COMPLETED' if gate5 else 'NOT RUN / MISSING'}",
            "",
        ]
    )

    if stage2 is not None:
        lines.extend(
            [
                "## Stage 1/2 5-fold validation",
                "",
                f"- RMSE: {stage2['rmse_mean']:.3f} +/- {stage2['rmse_std']:.3f}",
                f"- NASA: {stage2['nasa_mean']:.3f} +/- {stage2['nasa_std']:.3f}",
                f"- MAE: {stage2['mae_mean']:.3f} +/- {stage2['mae_std']:.3f}",
                f"- Folds: {stage2['folds']}",
                f"- Test files accessed: {stage2.get('test_files_accessed')}",
                "",
                "| Fold | RMSE | NASA | MAE |",
                "|---|---:|---:|---:|",
            ]
        )
        for fold, rmse, nasa, mae in zip(
            stage2["folds"],
            stage2["per_fold_rmse"],
            stage2["per_fold_nasa"],
            pd.read_csv(phase / "stage2_bidirectional_5fold" / "per_fold_metrics.csv")["mae"],
        ):
            lines.append(f"| {fold} | {rmse:.3f} | {nasa:.3f} | {mae:.3f} |")
        lines.append("")
        if not gate2:
            lines.extend(
                [
                    "Stage 2 did not meet the required validation RMSE threshold. "
                    "Per the task protocol, the official test set remains sealed and "
                    "Stage 5 must not be executed.",
                    "",
                ]
            )

    if pareto is not None:
        lines.extend(
            [
                "## Stage 3 Pareto scan",
                "",
                "| gamma | eta | RMSE | NASA | Late ratio | Severe late |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in pareto.iterrows():
            lines.append(
                f"| {row['gamma']:g} | {row['eta']:g} | {row['rmse']:.3f} | "
                f"{row['nasa_score']:.3f} | {row['late_ratio']:.3f} | "
                f"{row['severe_late_ratio']:.3f} |"
            )
        lines.append("")

    if final_metrics is not None:
        metrics = final_metrics["ensemble_metrics"]
        lines.extend(
            [
                "## Stage 5 blind evaluation",
                "",
                f"- RMSE: {metrics['rmse']:.3f}",
                f"- NASA: {metrics['nasa_score']:.3f}",
                f"- MAE: {metrics['mae']:.3f}",
                f"- Late ratio: {metrics['late_ratio']:.3f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Integrity statement",
            "",
            "All reported numbers are read from generated artifacts. No expected "
            "direction of effect, target score, or favorable seed was inserted by hand. "
            "If a gate failed, the corresponding downstream claim is left unsupported.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
