"""Plot the learned 14x14 sensor-space couplings (P10)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model import CRGraphTransformer  # noqa: E402

SENSORS = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc"]


def main() -> None:
    ckpt = ROOT / "experiments/p10/final2/FD001/p10_full_s2027/seed_2027/p10_full/final_model.pt"
    out = ROOT / "figures/final"
    out.mkdir(parents=True, exist_ok=True)
    model = CRGraphTransformer(num_sensors=14, window_size=50, d_model=88, num_heads=4,
                               num_transformer_layers=2, dropout=0.15, ff_mult=1.6,
                               use_crgcn=True, native_graph=True, tail_mode="adaptive")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    bases = model.spatial.base_matrices().detach().numpy()
    with torch.no_grad():
        cond = torch.zeros(1, 50, 3)
        pi = torch.softmax(model.spatial.router(cond), dim=-1)[0, 0].numpy()
    mixed = np.einsum("k,knm->nm", pi, bases)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [(bases[0], "Base graph A1"), (bases[1], "Base graph A2"),
              (bases[2], "Base graph A3"), (mixed, "Routed mixture (FD001 cond=0)")]
    for ax, (mat, title) in zip(axes, panels):
        im = ax.imshow(mat, cmap="viridis")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(14)); ax.set_yticks(range(14))
        ax.set_xticklabels(SENSORS, rotation=90, fontsize=6)
        ax.set_yticklabels(SENSORS, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("P10 native sensor-space couplings (hollow 14x14, row-stochastic)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "p10_sensor_graph.png", dpi=220)
    fig.savefig(out / "p10_sensor_graph.pdf")
    print("wrote", out / "p10_sensor_graph.png")


if __name__ == "__main__":
    main()
