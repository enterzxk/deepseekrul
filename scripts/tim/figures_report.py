"""Report figures for the TIM progress deck.

Reads the same JSON artefacts as the manuscript tables so that every number on a
slide is traceable to the final protocol, not to the pre-audit ledger.
"""
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

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "RUL区间测量学汇报" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1E4FA8"
MID = "#3D7BD9"
LIGHT = "#8FB3E0"
RED = "#D9534F"
GREY = "#8B97A8"
INK = "#1A2230"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.edgecolor": "#B8C2CC",
    "axes.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

SUBSETS = list(C.SUBSETS)


def load(name):
    return json.loads((ROOT / "experiments" / "tim" / name).read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------- convention
def fig_convention():
    m = load("measure.json")["convention"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(SUBSETS))
    g90 = [m[s]["0.90"]["gap"] for s in SUBSETS]
    g50 = [m[s]["0.50"]["gap"] for s in SUBSETS]
    w = 0.34
    b1 = ax.bar(x - w / 2, g90, w, color=BLUE, label="nominal 0.90")
    b2 = ax.bar(x + w / 2, g50, w, color=LIGHT, label="nominal 0.50")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"+{r.get_height():.3f}",
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=11, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBSETS, fontsize=13)
    ax.set_ylabel("integer endpoint $-$ randomised PIT", fontsize=13)
    ax.set_ylim(0, 0.138)
    ax.legend(fontsize=12, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_convention")


# --------------------------------------------------------------- detectability
def fig_detect():
    ns = np.unique(np.round(np.logspace(1.0, 3.3, 260)).astype(int))
    ds = [C.mde(int(n), 0.90, alpha=0.05 / 5, power=0.80) for n in ns]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvspan(11, 82, color=RED, alpha=0.09, lw=0,
               label="one fleet's bin (11–82)")
    ax.axvspan(84, 212, color="#E0A030", alpha=0.12, lw=0,
               label="pooled, four fleets (84–212)")
    ax.plot(ns, ds, color=BLUE, lw=2.0, label="detectable deviation $\\delta^\\star$")
    ax.axhline(0.10, color=RED, ls=":", lw=1.4)
    ax.axvline(143, color=RED, ls=":", lw=1.0, alpha=0.7)
    ax.annotate("0.10", (12, 0.108), fontsize=12, color=RED)
    ax.annotate("143 engines / bin", (150, 0.125), fontsize=12, color=RED)
    ax.set_xscale("log")
    ax.set_xlim(10, 2000)
    ax.set_ylim(0, 0.46)
    ax.set_xlabel("engines in the bin", fontsize=13)
    ax.set_ylabel("detectable deviation $\\delta^\\star$", fontsize=13)
    ax.legend(fontsize=10.5, loc="upper right")
    save(fig, "fig_detect")


# -------------------------------------------------------------------- object
def fig_object():
    o = load("object.json")["subsets"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(SUBSETS))
    single = [o[s]["single-1|P15|0.90"]["width"] for s in SUBSETS]
    matched = [o[s]["matched-3|P15|0.90"]["width"] for s in SUBSETS]
    w = 0.34
    b1 = ax.bar(x - w / 2, single, w, color=LIGHT, label="calibrate one fold model (CV+)")
    b2 = ax.bar(x + w / 2, matched, w, color=BLUE, label="calibrate the deployed ensemble")
    for i, (a, b) in enumerate(zip(single, matched)):
        # percentages are computed from the widths as reported in the manuscript table
        ch = 100 * (round(b, 1) / round(a, 1) - 1)
        ax.annotate(f"{ch:+.1f}%", (i, max(a, b) + 1.2), ha="center", fontsize=13,
                    color=RED if ch < -8 else INK)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBSETS, fontsize=13)
    ax.set_ylabel("mean interval width (cycles)", fontsize=13)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=10.5, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_object")


# ---------------------------------------------------------------------- tail
def fig_tail():
    d = load("final.json")["subsets"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(SUBSETS))
    saves = []
    for s in SUBSETS:
        a = d[s]["matched free@0.90"]["width"]
        b = d[s]["matched equal tails@0.90"]["width"]
        saves.append(100 * (1 - a / b))
    colors = [BLUE if v >= 0 else RED for v in saves]
    bars = ax.bar(x, saves, 0.46, color=colors)
    for r, v in zip(bars, saves):
        off = 0.5 if v >= 0 else -1.3
        ax.annotate(f"{v:+.1f}%", (r.get_x() + r.get_width() / 2, v + off),
                    ha="center", fontsize=13, color=INK)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBSETS, fontsize=13)
    ax.set_ylabel("width saved by a free tail (%)", fontsize=13)
    ax.set_ylim(-9, 9)
    ax.grid(axis="x", visible=False)
    ax.annotate("sign reverses", (3, -6.8), ha="center", fontsize=12, color=RED)
    save(fig, "fig_tail")


if __name__ == "__main__":
    fig_convention()
    fig_detect()
    fig_object()
    fig_tail()
    print("written to", OUT)
