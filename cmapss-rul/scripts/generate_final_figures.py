"""Generate the Phase-5 SCI figures from frozen predictions/checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from dataset import SELECTED_SENSOR_COLUMNS, load_cmapss_files  # noqa: E402
from model import STG_MambaNet  # noqa: E402

LABELS = ["S2", "S3", "S4", "S7", "S8", "S9", "S11", "S12", "S13", "S14", "S15", "S17", "S20", "S21"]


def save(fig, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_model(variant_dir: Path, device: torch.device):
    config = json.loads((variant_dir / "config.json").read_text(encoding="utf-8"))
    model = STG_MambaNet(**config["model_config"]).to(device)
    state = torch.load(
        variant_dir / "best_model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model, config


def lifecycle(model, config, variant_dir, data_dir, device):
    stats = json.loads((variant_dir / "scaler_stats.json").read_text(encoding="utf-8"))
    columns = list(SELECTED_SENSOR_COLUMNS)
    lower = np.asarray([stats[c][0] for c in columns], dtype=np.float64)
    upper = np.asarray([stats[c][1] for c in columns], dtype=np.float64)
    span = np.where(upper > lower, upper - lower, 1.0)
    _, test_frame, test_rul = load_cmapss_files(data_dir, "FD001")
    test_frame = test_frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    units = sorted(int(x) for x in test_frame["unit"].unique())
    lengths = np.asarray([int((test_frame["unit"] == u).sum()) for u in units])
    selected = np.argsort(lengths)[-4:]
    window = int(config["model_config"]["window_size"])
    out = {}
    for idx in selected:
        unit = units[int(idx)]
        raw = test_frame.loc[test_frame["unit"] == unit, columns].to_numpy(float)
        sensors = ((raw - lower) / span).astype(np.float32)
        n = sensors.shape[0]
        ends = np.arange(window - 1, n)
        windows = np.stack([sensors[e - window + 1 : e + 1] for e in ends])
        preds = []
        with torch.no_grad():
            for start in range(0, len(windows), 64):
                batch = torch.from_numpy(windows[start : start + 64]).to(device)
                preds.append(model(batch).cpu().numpy().reshape(-1))
        true = np.minimum(125.0, float(test_rul[int(idx)]) + (n - 1 - ends))
        out[unit] = (ends + 1, true, np.concatenate(preds))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fd001-root", type=Path, default=ROOT / "experiments" / "final" / "FD001_locked")
    p.add_argument("--fd002-root", type=Path, default=ROOT / "experiments" / "final" / "FD002_3seeds")
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "final")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    full_dir = args.fd001_root / "seed_101" / "seed_101" / "full"
    mse_dir = args.fd001_root / "seed_101" / "seed_101" / "wo_physics"
    if not full_dir.is_dir():
        raise FileNotFoundError(full_dir)

    pred = pd.read_csv(full_dir / "test_predictions.csv").sort_values("y_true")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(pred["y_true"].to_numpy(), color="black", lw=2, label="True RUL")
    ax.plot(pred["y_pred"].to_numpy(), color="#d1495b", lw=1.6, label="Predicted")
    ax.fill_between(
        np.arange(len(pred)),
        pred["y_true"].to_numpy(),
        pred["y_pred"].to_numpy(),
        color="#d1495b",
        alpha=0.12,
    )
    ax.set_xlabel("Test engine (sorted by true RUL)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title("FD001 test RUL (clip125, seed 101)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save(fig, args.out, "fig1_fd001_sorted_rul")

    full_model, full_cfg = load_model(full_dir, device)
    curves_full = lifecycle(full_model, full_cfg, full_dir, ROOT / "data" / "CMAPSS", device)
    if mse_dir.is_dir():
        mse_model, mse_cfg = load_model(mse_dir, device)
        curves_mse = lifecycle(mse_model, mse_cfg, mse_dir, ROOT / "data" / "CMAPSS", device)
    else:
        curves_mse = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, unit in zip(axes.ravel(), list(curves_full)[:4]):
        cycle, true, pred_full = curves_full[unit]
        ax.plot(cycle, true, color="black", lw=1.8, label="True")
        ax.plot(cycle, pred_full, color="#d1495b", lw=1.2, label="Full")
        if unit in curves_mse:
            ax.plot(cycle, curves_mse[unit][2], color="#3a6ea5", lw=1.2, label="Pure MSE")
        ax.set_title(f"Engine #{unit}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("FD001 lifecycle tracking: Full vs Pure MSE")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(args.out / "fig2_engine_trajectories.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out / "fig2_engine_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)

    fd002_full = args.fd002_root / "seed_101" / "seed_101" / "full"
    if fd002_full.is_dir():
        m1, _ = load_model(full_dir, device)
        m2, _ = load_model(fd002_full, device)
        a1 = m1.get_adjacency_matrix(detach=True).cpu().numpy()
        a2 = m2.get_adjacency_matrix(detach=True).cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for ax, mat, title in (
            (axes[0], a1, "FD001 (single condition)"),
            (axes[1], a2, "FD002 (6 conditions, KMeans norm)"),
        ):
            im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=max(a1.max(), a2.max()))
            ax.set_xticks(range(14))
            ax.set_yticks(range(14))
            ax.set_xticklabels(LABELS, rotation=45, ha="right")
            ax.set_yticklabels(LABELS)
            ax.set_title(title)
        fig.subplots_adjust(
            left=0.06, right=0.88, bottom=0.18, top=0.88, wspace=0.25
        )
        cax = fig.add_axes([0.91, 0.15, 0.015, 0.70])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Adjacency weight")
        fig.savefig(
            args.out / "fig3_learned_adjacency_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            args.out / "fig3_learned_adjacency_comparison.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    summary = pd.read_csv(ROOT / "results" / "final_FD001_5seeds" / "summary.csv")
    labels = [x.replace(" (matched LSTM)", "\n(matched LSTM)").replace(" (A=I)", "\n(A=I)") for x in summary["variant_label"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col, title in zip(
        axes.ravel(),
        ("rmse_mean", "nasa_mean", "late_fraction_mean", "mono_violation_rate_mean"),
        ("RMSE", "NASA Score", "Late Fraction", "Mono Violation Rate"),
    ):
        values = summary[col].to_numpy()
        bars = ax.bar(range(len(values)), values, color="#3a6ea5")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("FD001 five-seed ablation: accuracy and safety diagnostics")
    save(fig, args.out, "fig4_ablation_metrics_bar")
    print("figures written to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
