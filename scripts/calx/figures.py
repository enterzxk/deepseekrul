"""Figure for CONFORMAL_RESULTS.md: the coverage-width plane, and the width gain.

Left panel: every method at nominal 90%, one point per (subset, method). The shaded band is
+/-1 bootstrap SE around 0.90, so points inside it are indistinguishable from nominal at
that sample size. Right panel: paired bootstrap gain of the signed-residual construction
over the symmetric band.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
STYLE = {
    "base": ("#1f4e79", "o", "frozen layer"),
    "cv+": ("#c0504d", "s", "CV+ (symmetric)"),
    "ncv+": ("#e0a36a", "^", "normalised CV+"),
    "sym": ("#a33b3b", "D", "symmetric, mean-centred"),
    "asym/sig": ("#2e7d32", "*", "signed-residual, normalised"),
}


def main():
    c = json.load(open(ROOT / "experiments" / "calx" / "conformal.json"))["subsets"]
    a = json.load(open(ROOT / "experiments" / "calx" / "asym.json"))
    b = json.load(open(ROOT / "experiments" / "calx" / "bootstrap.json"))

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=160,
                           gridspec_kw={"width_ratios": [1.25, 1]})

    for s in SUBSETS:
        n = c[s]["n_test"]
        se = np.sqrt(0.9 * 0.1 / n)
        ax[0].axvspan(0, 100, color="#eeeeee", alpha=0.0)
        pts = {
            "base": (c[s]["methods"]["base"]["0.90"]["width"], c[s]["methods"]["base"]["0.90"]["cov"]),
            "cv+": (c[s]["methods"]["cv+"]["0.90"]["width"], c[s]["methods"]["cv+"]["0.90"]["cov"]),
            "ncv+": (c[s]["methods"]["ncv+"]["0.90"]["width"], c[s]["methods"]["ncv+"]["0.90"]["cov"]),
            "sym": (a[s]["sym@0.90"]["width"], a[s]["sym@0.90"]["cov"]),
            "asym/sig": (a[s]["asym/sig@0.90"]["width"], a[s]["asym/sig@0.90"]["cov"]),
        }
        for k, (w, cv) in pts.items():
            col, mk, lab = STYLE[k]
            ax[0].scatter(w, cv, c=col, marker=mk, s=70 if k != "asym/sig" else 160,
                          label=lab if s == "FD001" else None, zorder=3,
                          edgecolor="white", linewidth=0.6)
            ax[0].annotate(s, (w, cv), textcoords="offset points", xytext=(5, -3),
                           fontsize=6.5, color="#555555")

    ax[0].axhline(0.90, color="#333333", lw=1.1, ls="--", zorder=1)
    ax[0].fill_between([0, 100], 0.90 - 0.031, 0.90 + 0.031, color="#333333", alpha=0.08, zorder=0)
    ax[0].set_xlim(24, 64)
    ax[0].set_ylim(0.82, 1.0)
    ax[0].set_xlabel("mean interval width at nominal 90% (RUL cycles)")
    ax[0].set_ylabel("randomised empirical coverage")
    ax[0].set_title("Every method at 90%, all four fleets", fontsize=10, loc="left")
    ax[0].legend(fontsize=7, frameon=False, loc="upper left")
    ax[0].grid(alpha=0.25, lw=0.5)

    xpos = np.arange(len(SUBSETS))
    gain = [b[s]["width_gain"][0] for s in SUBSETS]
    top = [b[s]["width_gain"][1] for s in SUBSETS]
    mid = [b[s]["width_gain"][0] + (b[s]["width_gain"][1] - b[s]["width_gain"][0]) / 2 for s in SUBSETS]
    ax[1].bar(xpos, mid, color="#2e7d32", alpha=0.8, width=0.55)
    ax[1].errorbar(xpos, mid, yerr=[np.array(mid) - np.array(gain), np.array(top) - np.array(mid)],
                   fmt="none", ecolor="#222222", capsize=4, lw=1.1)
    ax[1].set_xticks(xpos, SUBSETS, fontsize=8)
    ax[1].set_ylabel("width saved vs symmetric band (cycles, bootstrap estimate)")
    ax[1].set_title("Signed-residual minus symmetric, 95% paired bootstrap", fontsize=10, loc="left")
    ax[1].grid(axis="y", alpha=0.25, lw=0.5)
    for i, v in enumerate(mid):
        ax[1].text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=7)

    fig.tight_layout()
    out = ROOT / "figures" / "calx"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "coverage_width.png", bbox_inches="tight")
    print("wrote", out / "coverage_width.png")


if __name__ == "__main__":
    main()
