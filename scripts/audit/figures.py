"""Figures for the coverage audit.

fig1  worst-band coverage per cell, with the bootstrap interval and the simultaneous
      lower bound. The point is visual: every interval crosses far below the nominal line.
fig2  detectability curve. delta* against per-band n, with the 21 audited cells placed on
      it. Everything sits in the region where delta* exceeds the effect anyone reports.
fig3  the pooled case: five bands, point estimate, simultaneous bound, nominal level.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import mde  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "font.family": "DejaVu Sans"})


def load(p):
    return json.loads(Path(p).read_text())


def fig1():
    a = load(ROOT / "experiments/cal/audit/audit_cmapss.json")
    b = load(ROOT / "experiments/cal/audit/audit_ncmapss.json")
    keys = list(a) + list(b)
    rows = []
    for k in list(a):
        v = a[k]
        rows.append((k, v["worst_picp"], v["lower_bound_worst"], v["p_marginal"], "C"))
    for k in list(b):
        v = b[k]
        rows.append((k, v["worst_picp"], v["lower_bound_worst"], v["p_marginal"], "N"))

    fig, ax = plt.subplots(figsize=(3.4, 4.4))
    y = np.arange(len(rows))[::-1]
    for yi, (k, p, lb, pm, tag) in zip(y, rows):
        col = "#1f4e79" if tag == "C" else "#a33a2a"
        ax.plot([lb, p], [yi, yi], color=col, lw=1.2, alpha=0.55, zorder=2)
        ax.plot([p], [yi], "o", ms=3.4, color=col, zorder=3)
        ax.plot([lb], [yi], "|", ms=6, color=col, zorder=3)
    ax.axvline(0.90, color="k", ls="--", lw=0.7, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_xlim(-0.02, 1.04)
    ax.set_xlabel("worst band: coverage (dot), 95% simultaneous lower bound (tick)",
                  fontsize=6.8)
    ax.text(0.905, len(rows) - 0.6, "nominal 0.90", fontsize=6, rotation=90,
            va="top", ha="left")
    ax.set_title("Every audited cell sits below nominal\nat its weakest band", fontsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.90, bottom=0.15)
    fig.savefig(OUT / "coverage_audit_fig1.png", dpi=220)
    plt.close(fig)


def fig2():
    ns = np.unique(np.round(np.geomspace(6, 420, 55)).astype(int))
    for alpha, lab, col, ls in ((0.05, "uncorrected (a=0.05)", "#7a7a7a", "--"),
                                (0.01, "Bonferroni, 5 bands", "#1f4e79", "-")):
        ds = [mde(int(n), 0.90, alpha=alpha) or np.nan for n in ns]
        plt.plot(ns, ds, ls, color=col, lw=1.1, label=lab)

    pts = []
    a = load(ROOT / "experiments/cal/audit/audit_cmapss.json")
    b = load(ROOT / "experiments/cal/audit/audit_ncmapss.json")
    for k, v in list(a.items()) + list(b.items()):
        pts.append((v["min_band_n_eff"], v["max_mde"], "C" if k.startswith("FD") else "N"))
    for n, d, t in pts:
        plt.plot([max(n, 2)], [d], "o", ms=3,
                 color="#1f4e79" if t == "C" else "#a33a2a", alpha=0.75)

    plt.axhline(0.10, color="#c0392b", lw=0.8, ls=":")
    plt.text(300, 0.108, "deficit of practical interest", fontsize=6, color="#c0392b")
    plt.axvline(72, color="#c0392b", lw=0.6, ls=":")
    plt.text(74, 0.45, "n=72", fontsize=6, color="#c0392b")
    plt.xscale("log")
    plt.xlabel("engines in the thinnest band (effective)")
    plt.ylabel("minimum detectable deficit  $\\delta^*$")
    plt.ylim(0, 0.9)
    plt.legend(fontsize=6, frameon=False, loc="upper right")
    plt.title("All 21 cells lie where the audit cannot resolve 0.10", fontsize=7.5)
    for s in ("top", "right"):
        plt.gca().spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "coverage_audit_fig2.png", dpi=220)
    plt.close()


def fig3():
    from scipy import stats as st
    r = load(ROOT / "experiments/cal/audit/audit_pooled.json")
    bands = [b["band"] for b in r["bands"]]
    picp = np.array([b["picp"] for b in r["bands"]])
    lo = np.array([st.beta.ppf(0.025, b["k"], b["n"] - b["k"] + 1) if b["k"] > 0 else 0.0
                   for b in r["bands"]])
    hi = np.array([st.beta.ppf(0.975, b["k"] + 1, b["n"] - b["k"]) if b["k"] < b["n"] else 1.0
                   for b in r["bands"]])
    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.vlines(x, lo, hi, color="#1f4e79", lw=1.6, alpha=0.45)
    ax.plot(x, picp, "o", ms=4.5, color="#1f4e79", label="band coverage")
    ax.axhline(r["a_hat"], color="#c0392b", lw=0.9, ls="--",
               label=f"marginal {r['a_hat']:.3f}")
    ax.axhline(0.90, color="k", lw=0.7, ls=":")
    ax.text(-0.42, 0.902, "nominal", fontsize=5.8, va="bottom")
    ax.annotate("[0.740, 0.904]", xy=(2, 0.904), xytext=(2.35, 0.80),
                fontsize=5.8, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=6.2)
    ax.set_ylim(0.68, 1.0)
    ax.set_ylabel("coverage (95% interval)")
    ax.set_title("653 engines: no band can be shown\nto fall below nominal", fontsize=7.5)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "coverage_audit_fig3.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote", [p.name for p in sorted(OUT.glob("coverage_audit_fig*.png"))])
