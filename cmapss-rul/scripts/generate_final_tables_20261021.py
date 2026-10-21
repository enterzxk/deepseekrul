"""Generate final tables and summary from frozen CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fmt(x, d=3):
    try:
        x = float(x)
    except Exception:
        return "--"
    return f"{x:.{d}f}" if np.isfinite(x) else "--"


def ms(mean, std, d=3):
    return f"{fmt(mean,d)} $\\pm$ {fmt(std,d)}"


def ablation_tex(df, caption, label):
    lines = [
        "\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}",
        f"\\label{{{label}}}", "\\begin{tabular}{lrrrrrrr}", "\\toprule",
        "Variant & RMSE & NASA & MAE & Late & Severe & Params \\\\", "\\midrule",
    ]
    order = ["wo_agcn", "identity_graph", "wo_ssm", "wo_physics", "wo_mono", "full"]
    indexed = df.set_index("variant")
    for key in order:
        if key not in indexed.index:
            continue
        r = indexed.loc[key]
        lines.append(
            f"{r['variant_label']} & {ms(r['rmse_mean'],r['rmse_std'])} & "
            f"{ms(r['nasa_mean'],r['nasa_std'])} & {fmt(r['mae_mean'])} & "
            f"{fmt(r['late_fraction_mean'])} & {fmt(r['severe_late_count_gt10_mean'],1)} & "
            f"{int(r['params']):,} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def table_tex(df, title, label, cols):
    headers = " & ".join(c[0] for c in cols)
    lines = [
        "\\begin{table*}[t]", "\\centering", f"\\caption{{{title}}}",
        f"\\label{{{label}}}", "\\begin{tabular}{" + cols[0][2] + "}", "\\toprule",
        headers + " \\\\", "\\midrule",
    ]
    for _, r in df.iterrows():
        values = []
        for _, key, _ in cols[1:]:
            values.append(key(r))
        lines.append(" & ".join(values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    fd001 = pd.read_csv(root / "results/final_FD001_5seeds/summary.csv")
    fd002 = pd.read_csv(root / "results/final_FD002_3seeds/summary.csv")
    ens = pd.read_csv(root / "results/final_ensemble_and_pvalues.csv")
    ext_path = root / "results/external_baselines_FD001_FD002.csv"
    ext = pd.read_csv(ext_path) if ext_path.is_file() else pd.DataFrame()
    unc_path = root / "results/uncertainty_frozen_FD001.csv"
    unc = pd.read_csv(unc_path) if unc_path.is_file() else pd.DataFrame()
    rob_path = root / "results/robustness_stress_FD001.csv"
    rob = pd.read_csv(rob_path) if rob_path.is_file() else pd.DataFrame()

    parts = [
        "% Auto-generated from frozen metrics. Do not hand-edit.",
        ablation_tex(fd001, "FD001 five-seed confirmatory ablation (clip125).", "tab:fd001"),
        ablation_tex(fd002, "FD002 three-seed external validation (clip125).", "tab:fd002"),
    ]
    ens_tex = table_tex(
        ens,
        "Multi-seed ensemble and paired Wilcoxon comparison.",
        "tab:ensemble",
        [
            ("l", None, "l"),
            ("Dataset", lambda r: r["subset"], "l"),
            ("Variant", lambda r: r["variant_label"], "l"),
            ("Seeds", lambda r: str(int(r["n_seeds"])), "r"),
            ("RMSE", lambda r: fmt(r["rmse"]), "r"),
            ("NASA", lambda r: fmt(r["nasa_score"]), "r"),
            ("MAE", lambda r: fmt(r["mae"]), "r"),
            ("Late", lambda r: fmt(r["late_ratio"]), "r"),
            ("p_abs", lambda r: fmt(r["p_abs_error_vs_full"], 4), "r"),
        ],
    )
    ens_tex = ens_tex.replace("\\begin{tabular}{l", "\\begin{tabular}{ll").replace("Late & p_abs \\\\", "Late & p_abs \\\\")
    # Build ensemble table explicitly to guarantee header count.
    ens_lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Multi-seed ensemble and paired Wilcoxon comparison.}",
        "\\label{tab:ensemble}", "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Dataset & Variant & Seeds & RMSE & NASA & MAE & Late & $p_{\\rm abs}$ \\\\",
        "\\midrule",
    ]
    for _, r in ens.iterrows():
        ens_lines.append(
            f"{r['subset']} & {r['variant_label']} & {int(r['n_seeds'])} & {fmt(r['rmse'])} & "
            f"{fmt(r['nasa_score'])} & {fmt(r['mae'])} & {fmt(r['late_ratio'])} & "
            f"{fmt(r['p_abs_error_vs_full'],4)} \\\\"
        )
    ens_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    parts.append("\n".join(ens_lines))

    if not ext.empty:
        ext_lines = [
            "\\begin{table*}[t]", "\\centering",
            "\\caption{External baselines under the frozen protocol.}",
            "\\label{tab:baselines}", "\\begin{tabular}{llrrrrrr}", "\\toprule",
            "Dataset & Baseline & RMSE & NASA & MAE & Late & Params & $p_{\\rm abs}$ \\\\",
            "\\midrule",
        ]
        for _, r in ext.iterrows():
            ext_lines.append(
                f"{r['subset']} & {r['baseline']} & {fmt(r['rmse'])} & {fmt(r['nasa_score'])} & "
                f"{fmt(r['mae'])} & {fmt(r['late_ratio'])} & {int(r['params']):,} & "
                f"{fmt(r.get('p_abs_error_vs_full', np.nan),4)} \\\\"
            )
        ext_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
        parts.append("\n".join(ext_lines))

    if not unc.empty:
        safe = unc[unc["record_type"] == "safe_bound"]
        lines = [
            "\\begin{table}[t]", "\\centering",
            "\\caption{MC-Dropout uncertainty and safe lower bound on FD001.}",
            "\\label{tab:uncertainty}", "\\begin{tabular}{rrrrrrr}", "\\toprule",
            "$k$ & RMSE & NASA & Late & Severe & PICP & MPIW \\\\", "\\midrule",
        ]
        for _, r in safe.iterrows():
            lines.append(
                f"{fmt(r['k'],2)} & {fmt(r['rmse'])} & {fmt(r['nasa_score'])} & "
                f"{fmt(r['late_fraction'])} & {int(r['severe_late_count_gt10'])} & "
                f"{fmt(r['PICP95'])} & {fmt(r['MPIW95'],2)} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        parts.append("\n".join(lines))

    if not rob.empty:
        pivot = rob.pivot_table(index="condition", columns="model", values="rmse", aggfunc="mean")
        models = list(pivot.columns)
        lines = [
            "\\begin{table*}[t]", "\\centering",
            "\\caption{FD001 noise and sensor-dropout stress test (RMSE).}",
            "\\label{tab:robustness}", "\\begin{tabular}{l" + "r"*len(models) + "}", "\\toprule",
            "Condition & " + " & ".join(models) + " \\\\", "\\midrule",
        ]
        for cond, r in pivot.iterrows():
            lines.append(cond + " & " + " & ".join(fmt(r[m]) for m in models) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
        parts.append("\n".join(lines))

    (root / "results/final_tables.tex").write_text("\n".join(parts), encoding="utf-8")

    full1 = fd001[fd001.variant == "full"].iloc[0]
    full2 = fd002[fd002.variant == "full"].iloc[0]
    ens1 = ens[(ens.subset == "FD001") & (ens.variant == "full")].iloc[0]
    ens2 = ens[(ens.subset == "FD002") & (ens.variant == "full")].iloc[0]
    md = [
        "# FINAL SUMMARY 2026-10-21",
        "",
        "## Headline",
        "",
        "| Protocol | RMSE | NASA | MAE |",
        "|---|---:|---:|---:|",
        f"| FD001 Full, 5-seed mean | {ms(full1.rmse_mean,full1.rmse_std)} | {ms(full1.nasa_mean,full1.nasa_std)} | {fmt(full1.mae_mean)} |",
        f"| FD001 Full, 5-seed ensemble | {fmt(ens1.rmse)} | {fmt(ens1.nasa_score)} | {fmt(ens1.mae)} |",
        f"| FD002 Full, 3-seed mean | {ms(full2.rmse_mean,full2.rmse_std)} | {ms(full2.nasa_mean,full2.nasa_std)} | {fmt(full2.mae_mean)} |",
        f"| FD002 Full, 3-seed ensemble | {fmt(ens2.rmse)} | {fmt(ens2.nasa_score)} | {fmt(ens2.mae)} |",
        "",
        "## Core scientific findings",
        "",
        "- A-GCN is the most consistent positive module; FD002 paired test is significant, FD001 effect size is consistent but not uniformly significant.",
        "- SSM beats LSTM on FD001 but loses to LSTM on FD002; this is an inductive-bias boundary.",
        "- Mono loss improves RMSE but raises late-prediction fraction and NASA score.",
        "- Multi-seed ensembles improve RMSE on both FD001 and FD002.",
        "- MC-Dropout uncertainty is calibrated in the sense of interval width and coverage; safe lower bounds reduce severe late predictions but can increase aggregate NASA score through early predictions.",
        "",
        "## Files",
        "",
        "- results/final_tables.tex",
        "- results/final_FD001_5seeds/summary.csv",
        "- results/final_FD002_3seeds/summary.csv",
        "- results/final_ensemble_and_pvalues.csv",
        "- results/external_baselines_FD001_FD002.csv",
        "- results/uncertainty_frozen_FD001.csv",
        "- results/robustness_stress_FD001.csv",
        "- figures/final/",
        "- paper/main.tex",
    ]
    (root / "FINAL_SUMMARY_20261021.md").write_text("\n".join(md), encoding="utf-8")
    print("final tables and summary generated")


if __name__ == "__main__":
    main()
