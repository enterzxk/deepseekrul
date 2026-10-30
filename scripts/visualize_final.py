"""Paper figures for the frozen FD001 confirmatory run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import STG_MambaNet  # noqa: E402
from dataset import SELECTED_SENSOR_COLUMNS, load_cmapss_files  # noqa: E402

SENSOR_LABELS = ["S2", "S3", "S4", "S7", "S8", "S9", "S11", "S12", "S13", "S14", "S15", "S17", "S20", "S21"]


def _save(fig, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "CMAPSS")
    parser.add_argument("--subset", default="FD001")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.run_dir / "full" / "config.json").read_text(encoding="utf-8"))
    model = STG_MambaNet(**config["model_config"]).to(args.device)
    state = torch.load(
        args.run_dir / "full" / "best_model.pt",
        map_location=args.device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    adjacency = model.get_adjacency_matrix(detach=True).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(adjacency, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(SENSOR_LABELS)))
    ax.set_yticks(range(len(SENSOR_LABELS)))
    ax.set_xticklabels(SENSOR_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(SENSOR_LABELS)
    ax.set_xlabel("Source sensor")
    ax.set_ylabel("Target sensor")
    alpha = model.spatial_branch.graph_alpha
    alpha_text = "n/a" if alpha is None else f"{float(alpha):.3f}"
    ax.set_title(f"Learned sensor coupling (clip125, alpha={alpha_text})")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, args.output_dir, "fig3_sensor_graph_heatmap")

    predictions = pd.read_csv(args.run_dir / "full" / "test_predictions.csv")
    ordered = predictions.sort_values("y_true").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(ordered["y_true"], label="True RUL", color="black", linewidth=2.0)
    ax.plot(ordered["y_pred"], label="Predicted RUL", color="#d1495b", linewidth=1.8)
    ax.set_xlabel("Test engine (sorted by true RUL)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title("FD001 test RUL prediction (clip125, seed 101)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, args.output_dir, "fig1_sorted_rul_prediction")

    # Fig. 2: full-lifecycle tracking for four long test engines.
    stats = json.loads(
        (args.run_dir / "full" / "scaler_stats.json").read_text(encoding="utf-8")
    )
    columns = list(SELECTED_SENSOR_COLUMNS)
    lower = np.asarray([stats[c][0] for c in columns], dtype=np.float64)
    upper = np.asarray([stats[c][1] for c in columns], dtype=np.float64)
    span = np.where(upper > lower, upper - lower, 1.0)
    _, test_frame, test_rul = load_cmapss_files(args.data_dir, args.subset)
    test_frame = test_frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    units = sorted(int(x) for x in test_frame["unit"].unique())
    lengths = np.asarray(
        [int((test_frame["unit"] == unit).sum()) for unit in units], dtype=np.int64
    )
    selected = np.argsort(lengths)[-4:]
    window_size = int(config["model_config"]["window_size"])
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=False)
    for ax, unit_idx in zip(axes.ravel(), selected):
        unit = units[int(unit_idx)]
        group = test_frame.loc[test_frame["unit"] == unit]
        raw = group.loc[:, columns].to_numpy(dtype=np.float64)
        sensors = ((raw - lower) / span).astype(np.float32)
        n_cycles = sensors.shape[0]
        ends = np.arange(window_size - 1, n_cycles)
        windows = np.stack(
            [sensors[end - window_size + 1 : end + 1] for end in ends]
        ).astype(np.float32)
        predictions = []
        model.eval()
        with torch.no_grad():
            for start in range(0, windows.shape[0], 64):
                batch = torch.from_numpy(windows[start : start + 64]).to(args.device)
                predictions.append(model(batch).cpu().numpy().reshape(-1))
        predicted = np.concatenate(predictions)
        true = np.minimum(
            125.0,
            float(test_rul[int(unit_idx)]) + (n_cycles - 1 - ends),
        )
        cycles = ends + 1
        ax.plot(cycles, true, color="black", linewidth=1.8, label="True RUL")
        ax.plot(cycles, predicted, color="#d1495b", linewidth=1.4, label="Predicted")
        ax.set_title(f"Engine #{unit} (length {n_cycles})")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL (cycles)")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("FD001 lifecycle tracking (clip125, seed 101)", y=0.98)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(args.output_dir / "fig2_engine_lifecycle_tracking.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "fig2_engine_lifecycle_tracking.pdf", bbox_inches="tight")
    plt.close(fig)

    summary_path = args.summary or (PROJECT_ROOT / "results" / "final_FD001_5seeds" / "summary.csv")
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        labels = [
            label.replace(" (matched LSTM)", "\n(matched LSTM)").replace(" (A=I)", "\n(A=I)")
            for label in summary["variant_label"].tolist()
        ]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
        for ax, column, title, color in (
            (axes[0], "rmse_mean", "RMSE (lower is better)", "#3a6ea5"),
            (axes[1], "nasa_mean", "NASA score (lower is better)", "#e07a5f"),
        ):
            values = summary[column].to_numpy()
            bars = ax.bar(x, values, color=color)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        fig.suptitle("FD001 five-seed ablation (clip125, seeds 101-202-303-404-505)")
        _save(fig, args.output_dir, "fig4_ablation_comparison")

    print("figures written to", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
