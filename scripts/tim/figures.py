"""Figures for the manuscript. Agg backend, 200 dpi, vector PDF alongside."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C

OUT = C.ROOT / "figures" / "tim"
OUT.mkdir(parents=True, exist_ok=True)
INK = "#1b1b1b"
GRID = "#d9d9d9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#8a8a8a",
    "axes.linewidth": 0.7, "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
})

SUBSETS = list(C.SUBSETS)
METHODS = [
    ("layer", "parametric layer", "#111111", "s"),
    ("SC-sym", "split, symmetric", "#c0392b", "o"),
    ("SC-asym", "split, signed", "#e08e0b", "o"),
    ("CV+-sym", "CV+, symmetric", "#8e44ad", "^"),
    ("CV+-asym", "CV+, signed", "#16a085", "^"),
    ("EAC-sym", "EAC, symmetric", "#d35400", "D"),
    ("EAC-eq", "EAC, equal tails", "#7aa9d6", "D"),
    ("EAC", "EAC", "#1f6fb4", "D"),
    ("DE-Gauss", "deep ensemble $\\pm z\\sigma$", "#7f8c8d", "X"),
    ("QR", "quantile regression", "#aab7b8", "v"),
    ("CQR", "CQR", "#2a9d8f", "*"),
]


def load(name):
    return json.loads((C.ROOT / "experiments" / "tim" / name).read_text())


# ------------------------------------------------------------------ fig 1
def fig_coverage_width():
    d = load("matrix.json")["subsets"]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.1),
                             gridspec_kw=dict(height_ratios=[1.12, 1.0], hspace=0.30, wspace=0.14))
    for row, (xl, yl) in enumerate((((7, 60), (0.42, 1.0)), ((22, 46), (0.855, 0.965)))):
        for j, (ax, s) in enumerate(zip(axes[row], SUBSETS)):
            n = d[s]["layer@0.90"]["n"]
            se = np.sqrt(0.9 * 0.1 / n)
            if row == 0:
                ax.axhspan(0.90 - 1.96 * se, 0.90 + 1.96 * se, color="#bcd4ea", alpha=0.45,
                           zorder=0,
                           label="binomial 95% band" if j == 0 else None)
            for key, lab, col, mk in METHODS:
                k = f"{key}@0.90"
                if k not in d[s]:
                    continue
                v = d[s][k]
                ax.scatter(v["width"], v["cov"], s=30, marker=mk, color=col,
                           label=lab if (row, j) == (0, 0) else None, zorder=3,
                           edgecolor="white", linewidth=0.45)
            ax.axhline(0.90, color="#444", ls="--", lw=0.8, zorder=1)
            ax.set_title(f"{s}  (n={n})", fontsize=8.5)
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            if j:
                ax.tick_params(labelleft=False)
            if row == 1:
                ax.set_xlabel("mean width (cycles)", fontsize=7.4)
    axes[0, 0].set_ylabel("coverage\n(nominal 0.90)", fontsize=7.8)
    axes[1, 0].set_ylabel("coverage, zoomed", fontsize=7.4)
    axes[1, 0].text(0.03, 0.94, "same points, zoomed on the cluster", transform=axes[1, 0].transAxes,
                    fontsize=6.2, color="#555", va="top")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=6.4, bbox_to_anchor=(0.5, -0.10),
               handletextpad=0.15, columnspacing=0.9)
    fig.savefig(OUT / "fig1_coverage_width.png")
    fig.savefig(OUT / "fig1_coverage_width.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2
def fig_simulation():
    d = load("simulation.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ns = [50, 100, 200, 500, 2000]
    for ax, lv in zip(axes, (0.90, 0.50)):
        for lab, key, col, mk in (("symmetric", "SC-sym", "#c0392b", "o"),
                                  ("signed, equal tails", "SC-asym", "#e08e0b", "s"),
                                  ("signed, free tails", "EAC", "#1f6fb4", "D")):
            mu = [d[f"{key}|{n}|{lv:.2f}"]["cov"] for n in ns]
            se = [d[f"{key}|{n}|{lv:.2f}"]["cov_se"] for n in ns]
            ax.errorbar(ns, mu, yerr=se, marker=mk, color=col, lw=1.2, ms=3.5,
                        capsize=2, label=lab)
        ax.axhline(lv, color="#555", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("calibration engines")
        ax.set_title(f"nominal {lv:.2f}", fontsize=8.5)
    axes[0].set_ylabel("coverage (nominal 0.90)")
    axes[1].set_ylabel("coverage (nominal 0.50)")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.savefig(OUT / "fig2_simulation.png")
    fig.savefig(OUT / "fig2_simulation.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig_arms():
    rep = json.loads((C.ROOT / "experiments" / "tim" / "ncmapss.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    lo, up, labs = [], [], []
    for s in SUBSETS:
        dev, tst, layer = C.load_fleet(s)
        z = []
        for sd in C.SEEDS:
            r, p = [], []
            for f in range(5):
                yc = dev[(sd, f)][1]
                pc = np.mean([dev[(x, f)][2] for x in C.SEEDS], axis=0)
                r.append(yc - pc)
                p.append(pc)
            r = np.concatenate(r)
            p = np.concatenate(p)
            from run_matrix import fit_scale
            sig = fit_scale(p, r)
            z.append(r / sig(p))
        z = np.concatenate(z)
        lo.append(-np.quantile(z, 0.05))
        up.append(np.quantile(z, 0.95))
        labs.append(s)
    x = np.arange(len(labs))
    ax.bar(x - 0.19, lo, 0.36, color="#1f6fb4", label="lower arm, $-q_{0.05}$")
    ax.bar(x + 0.19, up, 0.36, color="#e08e0b", label="upper arm, $q_{0.95}$")
    top = max(max(lo), max(up))
    for i, (l, u) in enumerate(zip(lo, up)):
        ax.text(i, max(l, u) + 0.04 * top, f"x{l/u:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, top * 1.28)
    ax.set_ylabel("arm length, normalised score")
    ax.set_title("the two arms are not the same length", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax = axes[1]
    rows = []
    for lv in ("0.90", "0.50"):
        rows.append((lv, 1 - rep["pooled"][lv]["asym"]["width"] / rep["pooled"][lv]["sym"]["width"]))
    ax.bar([r[0] for r in rows], [r[1] * 100 for r in rows], color=["#1f6fb4", "#7aa9d6"], width=0.5)
    for i, r in enumerate(rows):
        ax.text(i, r[1] * 100 + 1, f"{r[1]*100:.0f}%", ha="center", fontsize=8)
    ax.set_ylabel("width saved (%)")
    ax.set_title("symmetry tax, N-CMAPSS pooled", fontsize=8.5)
    ax.set_ylim(0, 60)
    fig.savefig(OUT / "fig3_arms.png")
    fig.savefig(OUT / "fig3_arms.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 4
def fig_measure():
    m = load("measure.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    x = np.arange(len(SUBSETS))
    g90 = [m["convention"][s]["0.90"]["gap"] for s in SUBSETS]
    g50 = [m["convention"][s]["0.50"]["gap"] for s in SUBSETS]
    ax.bar(x - 0.19, g90, 0.36, color="#1f6fb4", label="nominal 0.90")
    ax.bar(x + 0.19, g50, 0.36, color="#e08e0b", label="nominal 0.50")
    ax.set_xticks(x)
    ax.set_xticklabels(SUBSETS)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("integer-endpoint $-$ randomised PIT")
    ax.set_title("coverage bias of the reporting convention", fontsize=8.5)
    ax.legend(fontsize=7)
    ax = axes[1]
    ns = np.unique(np.round(np.logspace(1.05, 3.3, 220)).astype(int))
    ds = [C.mde(n, 0.90, alpha=0.05 / 5, power=0.80) for n in ns]
    ax.plot(ns, ds, color="#1f6fb4", lw=1.4, label="detectable deviation")
    ax.axhline(0.10, color="#c0392b", ls=":", lw=1.0)
    ax.text(1250, 0.108, "10 points", fontsize=6.5, color="#c0392b", ha="right")
    ax.axvspan(11, 82, color="#c0392b", alpha=0.12, label="one fleet's bin (11-82)")
    ax.axvspan(84, 212, color="#e08e0b", alpha=0.16, label="pooled, four fleets (84-212)")
    ax.legend(loc="lower left", fontsize=6.4, handlelength=1.2)
    ax.set_xscale("log")
    ax.set_xlim(10, 2000)
    ax.set_ylim(0, 0.36)
    ax.set_xlabel("engines in the bin")
    ax.set_ylabel("detectable deviation $\\delta^\\star$")
    ax.set_title("what a bin of that size can prove", fontsize=8.5)
    fig.savefig(OUT / "fig4_measure.png")
    fig.savefig(OUT / "fig4_measure.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_coverage_width()
    fig_simulation()
    fig_arms()
    fig_measure()
    print("figures in", OUT)
