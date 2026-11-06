"""Render paired held-out engine errors and NASA tails from exported CCT evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--skill-root", type=Path, required=True)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.skill_root / "scripts"))
    from export_figure import export_figure
    from layout_tools import add_panel_labels
    from profile_data import profile_data
    from setup_style import setup_style
    from visual_qa import audit_layout, print_report, render_preview

    pairs = pd.read_csv(args.evidence / "paired_engines.csv")
    contrasts = pd.read_csv(args.evidence / "paired_comparisons.csv")
    stage = next(x for x in ("confirmation", "tail", "screen") if x in set(pairs.stage))
    pairs = pairs[pairs.stage == stage]
    comparisons = contrasts[contrasts.stage == stage]
    profile = profile_data(
        pairs[["subset", "reference", "seed", "delta_engine_rmse", "candidate_nasa", "reference_nasa"]],
        group_cols=["subset", "reference"],
    )
    (args.evidence / "figure_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )

    # A seed average in squared error space retains the experiment's RMSE definition.
    per_engine = pairs.groupby(["subset", "reference", "unit"], as_index=False).agg(
        candidate_mse=("candidate_mse", "mean"), reference_mse=("reference_mse", "mean"),
        candidate_nasa=("candidate_nasa", "mean"), reference_nasa=("reference_nasa", "mean"),
    )
    per_engine["delta_rmse"] = np.sqrt(per_engine.candidate_mse) - np.sqrt(per_engine.reference_mse)
    panel_data = {}
    for subset in ("FD002", "FD004"):
        references = comparisons.loc[comparisons.subset == subset, "reference"].tolist()
        baseline = references[0]
        fixed = next(x for x in references if x.startswith("fixed_trend"))
        selected = per_engine[per_engine.subset == subset]
        base_pairs = selected[selected.reference == baseline]
        fixed_pairs = selected[selected.reference == fixed]
        worst = base_pairs.assign(priority=base_pairs[["candidate_nasa", "reference_nasa"]].max(axis=1))
        panel_data[subset] = (base_pairs, fixed_pairs, worst.nlargest(5, "priority"))

    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
    plt.rcParams.update({
        "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.grid": False,
        "savefig.bbox": None,
    })
    size = (7.16, 4.8)
    fig, axes = plt.subplots(2, 2, figsize=size)
    fig.subplots_adjust(left=0.123, right=0.982, top=0.866, bottom=0.106,
                        wspace=0.34, hspace=0.64)
    blue, gray, orange = "#0072B2", "#555555", "#D55E00"
    delta_span = max(
        abs(frame.delta_rmse).max()
        for base_pairs, fixed_pairs, _ in panel_data.values()
        for frame in (base_pairs, fixed_pairs)
    )
    nasa = np.concatenate([
        top[["candidate_nasa", "reference_nasa"]].to_numpy().ravel()
        for _, _, top in panel_data.values()
    ])
    use_log = nasa.min() > 0 and nasa.max() / nasa.min() >= 10
    nasa_limits = ((nasa.min() / 1.25, nasa.max() * 1.25)
                   if use_log else (0.0, nasa.max() * 1.08))

    for column, (subset, (base_pairs, fixed_pairs, top)) in enumerate(panel_data.items()):
        ax = axes[0, column]
        for frame, color, linestyle in ((base_pairs, blue, "-"), (fixed_pairs, gray, "--")):
            delta = np.sort(frame.delta_rmse.to_numpy())
            percentile = 100 * np.arange(1, len(delta) + 1) / len(delta)
            ax.step(delta, percentile, where="post", color=color, linestyle=linestyle, lw=1.35)
        ax.axvline(0.0, color="#777777", lw=0.8, linestyle=":")
        ax.set(xlim=(-1.07 * max(delta_span, 1.0), 1.07 * max(delta_span, 1.0)), ylim=(0, 100),
               title=f"{subset} ($n$ = {len(base_pairs)})", xlabel="CCT − control engine RMSE (cycles)",
               ylabel="Held-out engines (%)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.5)

        ax = axes[1, column]
        row = np.arange(len(top))
        for y, a, b in zip(row, top.reference_nasa, top.candidate_nasa):
            ax.plot([a, b], [y, y], color="#b2b2b2", lw=1, zorder=1)
        ax.scatter(top.reference_nasa, row, marker="s", s=22, color=blue,
                   edgecolors="#252525", linewidths=0.35, zorder=3)
        ax.scatter(top.candidate_nasa, row, marker="o", s=26, color=orange,
                   edgecolors="#252525", linewidths=0.35, zorder=4)
        ax.set_yticks(row, [str(unit) for unit in top.unit])
        ax.invert_yaxis()
        if use_log:
            ax.set_xscale("log")
        else:
            ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set(xlim=nasa_limits, title=f"{subset} highest NASA penalties", ylabel="Engine ID",
               xlabel="Mean NASA penalty" + (" (log scale)" if use_log else ""))
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.5)

    add_panel_labels(fig, axes=axes.ravel(), style="ieee", fontsize=9, x_offset_pt=-14)
    handles = [
        Line2D([0], [0], color=blue, lw=1.35, label="RMSE vs window control"),
        Line2D([0], [0], color=gray, linestyle="--", lw=1.35, label="RMSE vs fixed trend"),
        Line2D([0], [0], color=blue, marker="s", linestyle="none", label="NASA window control"),
        Line2D([0], [0], color=orange, marker="o", linestyle="none", label="NASA CCT"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.994), ncol=4,
               frameon=False, columnspacing=1.15, handlelength=1.75)
    render_preview(fig, str(args.evidence / "cct_paired_evidence_preview.png"), dpi=150)
    print_report(audit_layout(fig))
    if args.export:
        from PIL import Image, ImageOps

        basename = args.evidence / "cct_paired_evidence"
        export_figure(fig, str(basename), formats=["pdf", "svg", "png"],
                      size_inches=size, dpi=600, tight=False)
        with Image.open(basename.with_suffix(".png")) as color:
            ImageOps.grayscale(color).save(
                args.evidence / "cct_paired_evidence_grayscale.png", dpi=(600, 600),
            )
    plt.close(fig)


if __name__ == "__main__":
    main()
