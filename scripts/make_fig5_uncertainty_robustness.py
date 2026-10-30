"""Figure 5: uncertainty envelope and robustness curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncertainty", type=Path, default=ROOT / "results" / "uncertainty_frozen_FD001.npz")
    parser.add_argument("--robustness", type=Path, default=ROOT / "results" / "robustness_stress_FD001.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "final")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.uncertainty)
    mean = data["mean"]
    std = data["std"]
    lower = data["lower95"]
    upper = data["upper95"]
    true = data["y_true"]
    order = np.argsort(true)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(true))[order]
    axes[0].plot(x, true[order], "k-", linewidth=2, label="True RUL")
    axes[0].plot(x, mean[order], color="#C44E52", linewidth=1.5, label="MC mean")
    axes[0].fill_between(x, lower[order], upper[order], color="#C44E52", alpha=0.2, label="95% interval")
    axes[0].set_xlabel("Test engines sorted by true RUL")
    axes[0].set_ylabel("RUL (cycles)")
    axes[0].set_title("Predictive uncertainty envelope")
    axes[0].legend(loc="upper left", fontsize=8)
    if args.robustness.is_file():
        df = pd.read_csv(args.robustness)
        noise = df[df["noise_sigma"] > 0]
        if not noise.empty:
            for model, group in noise.groupby("model"):
                group = group.sort_values("noise_sigma")
                axes[1].plot(group["noise_sigma"], group["rmse"], marker="o", label=model)
            axes[1].set_xlabel("Gaussian noise sigma")
            axes[1].set_ylabel("RMSE (cycles)")
            axes[1].set_title("Noise robustness")
            axes[1].legend(fontsize=8)
        else:
            axes[1].text(0.5, 0.5, "Robustness data empty", ha="center", va="center")
    else:
        axes[1].text(0.5, 0.5, "Robustness results pending", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig5_uncertainty_and_robustness.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "fig5_uncertainty_and_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
