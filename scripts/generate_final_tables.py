"""Generate results/final_tables.tex from frozen summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_tables.tex"


def esc(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%")


def main_table(frame: pd.DataFrame, caption: str, label: str) -> list[str]:
    best_rmse = frame["rmse_mean"].min()
    best_nasa = frame["nasa_mean"].min()
    best_mae = frame["mae_mean"].min()
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Variant & RMSE & NASA & MAE & Late & Severe & Mono & Params \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        rmse = f"{row['rmse_mean']:.2f} $\\pm$ {row['rmse_std']:.2f}"
        nasa = f"{row['nasa_mean']:.2f} $\\pm$ {row['nasa_std']:.2f}"
        mae = f"{row['mae_mean']:.2f}"
        if abs(row["rmse_mean"] - best_rmse) < 1e-12:
            rmse = f"\\textbf{{{rmse}}}"
        if abs(row["nasa_mean"] - best_nasa) < 1e-12:
            nasa = f"\\textbf{{{nasa}}}"
        if abs(row["mae_mean"] - best_mae) < 1e-12:
            mae = f"\\textbf{{{mae}}}"
        lines.append(
            f"{esc(row['variant_label'])} & {rmse} & {nasa} & {mae} & "
            f"{row['late_fraction_mean']:.3f} & "
            f"{row['severe_late_count_gt10_mean']:.1f} & "
            f"{row['mono_violation_rate_mean']:.3f} & "
            f"{int(row['params']):,} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return lines


def raw_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Appendix: raw (unclipped) FD001 test labels.}",
        "\\label{tab:fd001-raw}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Variant & RMSE & NASA \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{esc(row['variant_label'])} & "
            f"{row['rmse_mean']:.2f} $\\pm$ {row['rmse_std']:.2f} & "
            f"{row['nasa_mean']:.2f} $\\pm$ {row['nasa_std']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return lines


def stats_table(frame: pd.DataFrame, caption: str, label: str) -> list[str]:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Comparison & $\\Delta$ abs. error & 95\\% CI & Wilcoxon $p$ \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{esc(row['variant_label'])} & "
            f"{row['mean_delta_abs_error']:+.3f} & "
            f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] & "
            f"{row['wilcoxon_p']:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return lines


def main() -> int:
    parts = ["% Auto-generated from frozen metrics; do not hand-edit.", ""]
    fd001 = ROOT / "results" / "final_FD001_5seeds" / "summary.csv"
    if fd001.is_file():
        parts += main_table(
            pd.read_csv(fd001),
            "FD001 five-seed confirmatory ablation (clip125).",
            "tab:fd001-ablation",
        )
        parts += main_table(
            pd.read_csv(fd001),
            "FD001 safety and physical-compliance diagnostics (clip125).",
            "tab:fd001-safety",
        )
    raw = ROOT / "results" / "final_FD001_5seeds" / "summary_appendix_raw.csv"
    if raw.is_file():
        parts += raw_table(pd.read_csv(raw))
    fd002 = ROOT / "results" / "final_FD002_3seeds" / "summary.csv"
    if fd002.is_file():
        parts += main_table(
            pd.read_csv(fd002),
            "FD002 multi-condition external validity (clip125, seeds 101/202/303).",
            "tab:fd002-ablation",
        )
    p6 = ROOT / "results" / "final_FD001_P6_5seeds" / "summary.csv"
    if p6.is_file():
        parts += main_table(
            pd.read_csv(p6),
            "Phase 6 FD001 structural-head and hollow-graph run (new seeds 2027-2031).",
            "tab:p6-fd001",
        )
    p6_stats = ROOT / "results" / "final_FD001_P6_5seeds" / "paired_stats.csv"
    if p6_stats.is_file():
        parts += stats_table(
            pd.read_csv(p6_stats),
            r"Phase 6 paired per-engine tests against Full (negative $\Delta$ favours the comparator).",
            "tab:p6-stats",
        )
    p7 = ROOT / "results" / "final_FD001_P7_5seeds" / "summary.csv"
    if p7.is_file():
        parts += main_table(
            pd.read_csv(p7),
            "Phase 7 FD001: BiLSTM-Attn backbone, clean input, alpha=0.15 (seeds 2027-2031).",
            "tab:p7-fd001",
        )
    p7_fd2 = ROOT / "results" / "final_FD002_P7_1seed" / "summary.csv"
    if p7_fd2.is_file():
        parts += main_table(
            pd.read_csv(p7_fd2),
            "Phase 7 FD002 protocol check (seed 2027, KMeans condition normalization).",
            "tab:p7-fd002",
        )
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
