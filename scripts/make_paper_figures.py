"""Generate 300 DPI paper figures from saved prediction/analysis tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--per-fold", type=Path, required=True)
    parser.add_argument("--pareto", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    folds = pd.read_csv(args.per_fold)
    pareto = pd.read_csv(args.pareto)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(predictions["y_true"], predictions["y_ensemble"], s=24, alpha=0.75)
    low = min(predictions["y_true"].min(), predictions["y_ensemble"].min())
    high = max(predictions["y_true"].max(), predictions["y_ensemble"].max())
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("FD001 five-fold ensemble")
    save(fig, args.output_dir, "fig1_prediction_scatter")

    error = predictions["signed_error"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.hist(error, bins=24, alpha=0.8, edgecolor="black")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Signed error: prediction - truth (cycles)")
    ax.set_ylabel("Engine count")
    ax.set_title("Error distribution")
    save(fig, args.output_dir, "fig2_error_distribution")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.bar(folds["fold"].astype(str), folds["rmse"], color="#4C72B0")
    ax.set_xlabel("Validation fold")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("Fold-wise validation performance")
    save(fig, args.output_dir, "fig3_fold_rmse")

    per_unit_nasa = predictions.copy()
    delta = per_unit_nasa["y_ensemble"].to_numpy() - per_unit_nasa["y_true"].to_numpy()
    per_unit_nasa["nasa"] = np.where(delta < 0, np.expm1(-delta / 13), np.expm1(delta / 10))
    per_unit_nasa = per_unit_nasa.sort_values("nasa", ascending=False)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(np.arange(len(per_unit_nasa)), per_unit_nasa["nasa"], color="#C44E52")
    ax.set_xlabel("Test engine rank")
    ax.set_ylabel("NASA score")
    ax.set_title("Engine-level asymmetric risk")
    save(fig, args.output_dir, "fig4_engine_risk")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    scatter = axes[0].scatter(
        pareto["rmse"], pareto["nasa_score"], c=pareto["gamma"], cmap="viridis", s=55
    )
    for _, row in pareto.iterrows():
        axes[0].annotate(f"gamma={row['gamma']:g}", (row["rmse"], row["nasa_score"]), fontsize=8)
    axes[0].set_xlabel("Validation RMSE (cycles)")
    axes[0].set_ylabel("Validation NASA score")
    axes[0].set_title("RMSE vs asymmetric risk")
    fig.colorbar(scatter, ax=axes[0], label="gamma")
    axes[1].plot(pareto["late_ratio"], pareto["rmse"], marker="o")
    for _, row in pareto.iterrows():
        axes[1].annotate(f"{row['gamma']:g}", (row["late_ratio"], row["rmse"]), fontsize=8)
    axes[1].set_xlabel("Late prediction ratio")
    axes[1].set_ylabel("Validation RMSE (cycles)")
    axes[1].set_title("Late-risk trade-off")
    save(fig, args.output_dir, "fig5_pareto_risk_tradeoff")

    if args.ablation is not None and args.ablation.is_file():
        table = pd.read_csv(args.ablation)
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(table["variant"], table["rmse_mean"], yerr=table["rmse_std"], capsize=4)
        ax.set_ylabel("Test RMSE (cycles)")
        ax.set_title("Ablation study")
        ax.tick_params(axis="x", rotation=15)
        save(fig, args.output_dir, "fig6_ablation_rmse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
