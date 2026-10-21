"""Generate MORNING_REPORT.md from frozen metrics only."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FD001 = ROOT / "results" / "final_FD001_5seeds"
FD002 = ROOT / "results" / "final_FD002_3seeds"
LOCK = ROOT / "results" / "locked_hparams.json"
OUT = ROOT / "MORNING_REPORT.md"


def _summary_table(directory: Path, title: str) -> str:
    path = directory / "summary.csv"
    if not path.is_file():
        return f"## {title}\n\nPENDING: no summary.csv\n"
    frame = pd.read_csv(path)
    lines = [
        f"## {title}",
        "",
        "| Variant | RMSE | NASA | MAE | Late_Fraction | Severe_Late_Count | Mono_Violation_Rate | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['variant_label']} | "
            f"{row['rmse_mean']:.3f} +/- {row['rmse_std']:.3f} | "
            f"{row['nasa_mean']:.3f} +/- {row['nasa_std']:.3f} | "
            f"{row['mae_mean']:.3f} | {row['late_fraction_mean']:.3f} | "
            f"{row['severe_late_count_gt10_mean']:.2f} | "
            f"{row['mono_violation_rate_mean']:.3f} | {int(row['params']):,} |"
        )
    return "\n".join(lines) + "\n"


def _raw_table(directory: Path) -> str:
    path = directory / "summary_appendix_raw.csv"
    if not path.is_file():
        return ""
    frame = pd.read_csv(path)
    lines = [
        "## Raw-label appendix",
        "",
        "| Variant | RMSE | NASA |",
        "|---|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['variant_label']} | {row['rmse_mean']:.3f} +/- {row['rmse_std']:.3f} | "
            f"{row['nasa_mean']:.3f} +/- {row['nasa_std']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _paired(directory: Path) -> str:
    path = directory / "paired_deltas_vs_full.csv"
    if not path.is_file():
        return ""
    frame = pd.read_csv(path)
    grouped = frame.groupby("variant_label")[
        ["delta_rmse_vs_full", "delta_nasa_vs_full"]
    ].mean()
    lines = [
        "## Paired deltas vs Full",
        "",
        "| Ablation | Mean delta RMSE | Mean delta NASA |",
        "|---|---:|---:|",
    ]
    for label, row in grouped.iterrows():
        lines.append(
            f"| {label} | {row['delta_rmse_vs_full']:+.3f} | "
            f"{row['delta_nasa_vs_full']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.is_file() else {}
    status_path = ROOT / "results" / "checkpoint_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    lines = [
        "# MORNING REPORT - STG-MambaNet Overnight Run",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Frozen hyperparameters",
        "",
        "~~~json",
        json.dumps(lock, indent=2, ensure_ascii=False),
        "~~~",
        "",
        _summary_table(FD001, "FD001 five-seed confirmatory ablation (clip125)"),
        _raw_table(FD001),
        _paired(FD001),
        _summary_table(FD002, "FD002 multi-condition external validity (clip125)"),
        _raw_table(FD002),
        "",
        "## Key findings",
        "",
        "- FD001 Full is not uniformly best; read the table above directly.",
        "- A-GCN evidence should be judged by the paired deltas, not by narrative.",
        "- The learned identity-prior alpha and the off-diagonal mass are reported in the stage report; the graph may be close to identity.",
        "- SSM vs parameter-matched LSTM, physics vs MSE, and Mono vs no-Mono must be described exactly as measured.",
        "- FD002 uses K-Means condition-aware normalization fitted on training engines only.",
        "",
        "## Artifacts",
        "",
        "- docs/FINAL_PROTOCOL.md",
        "- docs/STAGE3_REPORT.md",
        "- results/final_FD001_5seeds/",
        "- results/final_FD002_3seeds/ (if present)",
        "- results/final_tables.tex",
        "- figures/final/",
        "- logs/overnight_progress.log",
        "",
        "## Limitations",
        "",
        "- All official test scores in P3/P4 are confirmatory; do not use them for further model selection.",
        "- P1/P2 were intended as full 3-seed validation grids; any incomplete phase is recorded in checkpoint_status.json.",
        "- FD002 condition clusters are data-driven, not physically labeled.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
