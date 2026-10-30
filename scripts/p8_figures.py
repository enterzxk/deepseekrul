"""P8 figures: accuracy bars, paired deltas, monotonicity panel."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "tcn": "TCN",
    "bilstm": "BiLSTM",
    "transformer": "Transformer",
    "p8_wo_graph": "w/o Graph",
    "p8_wo_phys_mono": "w/o Phys+Mono",
    "p8_full": "Ours Full",
}
COLORS = {
    "tcn": "#9aa5b1",
    "bilstm": "#9aa5b1",
    "transformer": "#6b7785",
    "p8_wo_graph": "#8fb8de",
    "p8_wo_phys_mono": "#5b8fc9",
    "p8_full": "#1f4e79",
}


def main() -> None:
    data = json.loads((ROOT / "results" / "p8" / "p8_stats.json").read_text())
    subsets = [s for s in ("FD001", "FD002") if s in data]

    # Figure 1: RMSE with seed-level spread
    fig, axes = plt.subplots(1, len(subsets), figsize=(6.2 * len(subsets), 4.2), squeeze=False)
    for ax, subset in zip(axes[0], subsets):
        rows = [r for r in data[subset] if r["variant"] in SHORT]
        names = [SHORT[r["variant"]] for r in rows]
        means = [r["rmse_mean"] for r in rows]
        stds = [r["rmse_std"] for r in rows]
        colors = [COLORS[r["variant"]] for r in rows]
        x = np.arange(len(rows))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
        for xi, r in zip(x, rows):
            vals = list(r["per_seed_rmse"].values())
            ax.scatter([xi] * len(vals), vals, s=14, color="black", alpha=0.55, zorder=3)
        ax.axhline(12.5, color="crimson", linestyle="--", linewidth=1.2, label="Q2 gate 12.5")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Test RMSE (cycles)")
        ax.set_ylim(bottom=max(0.0, min(means) - 3.0))
        ax.set_title(f"{subset}: 5-seed test RMSE")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p8_rmse_comparison.png", dpi=200)
    fig.savefig(OUT / "p8_rmse_comparison.pdf")
    plt.close(fig)

    # Figure 2: paired deltas vs full
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ys, labels, lows, highs, means = [], [], [], [], []
    for subset in subsets:
        stats = data.get("stats", {}).get(subset, {})
        for variant, st in stats.items():
            if variant not in SHORT:
                continue
            ys.append(f"{subset} {SHORT[variant]}")
            means.append(st["mean_delta"])
            lows.append(st["ci_low"])
            highs.append(st["ci_high"])
    ypos = np.arange(len(ys))
    err = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]
    ax.errorbar(means, ypos, xerr=err, fmt="o", color="#1f4e79", ecolor="#6b7785", capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ys, fontsize=8)
    ax.set_xlabel("Paired RMSE delta vs Ours Full (negative = ours better)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p8_paired_deltas.png", dpi=200)
    fig.savefig(OUT / "p8_paired_deltas.pdf")
    plt.close(fig)

    # Figure 3: monotonicity
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    width = 0.38
    for k, subset in enumerate(subsets):
        rows = [r for r in data[subset] if r["variant"] in SHORT]
        x = np.arange(len(rows)) + (k - 0.5) * width
        ax.bar(x, [r["mono_mean"] for r in rows], width, label=subset)
    ax.set_ylabel("Monotone violation rate")
    ax.set_xticks(np.arange(len([r for r in data[subsets[0]] if r["variant"] in SHORT])))
    ax.set_xticklabels([SHORT[r["variant"]] for r in data[subsets[0]] if r["variant"] in SHORT],
                       rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p8_monotonicity.png", dpi=200)
    fig.savefig(OUT / "p8_monotonicity.pdf")
    plt.close(fig)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
