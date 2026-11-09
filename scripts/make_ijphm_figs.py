"""Figures for the IJPHM v2 manuscript (measurement/evaluation paper).

Three figures, all from frozen artefacts, no manual numbers:
  fig1_effect_distribution : per-cell relative Winkler change, CQR vs constant
                             and MC-Dropout scale vs same-centre constant
  fig2_fold_separation     : fold-level scale effect, grouped by operating-
                             condition structure (multi vs single)
  fig3_engine_coverage     : per-engine coverage distribution of the frozen CQR

Runs where matplotlib is available (the a40 conda env).  Local Windows python
has no matplotlib; generate remotely and copy the PDFs back.

Usage:
    python scripts/make_ijphm_figs.py --root ~/deepseek-rul --out docs/figures/ijphm_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 7, 2024, 13]
MULTI = ["FD002", "FD004"]


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def cqr_per_engine(root: Path):
    rows = []
    for sub in SUBSETS:
        for fold in range(5):
            for seed in SEEDS:
                p = root / "experiments/punet_matrix" / sub / f"f{fold}_s{seed}" / "preds_cqr.csv"
                if not p.exists():
                    continue
                c = np.genfromtxt(p, delimiter=",", names=True)
                y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
                cal, tst = engine_split(c["unit"])
                s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
                q = float(np.quantile(s, min(1.0, 0.9 * (1 + 1 / s.size)), method="higher"))
                covered = (y[tst] >= lo[tst] - q) & (y[tst] <= hi[tst] + q)
                units = c["unit"][tst]
                for u in np.unique(units):
                    rows.append(dict(subset=sub, seed=seed, fold=fold, unit=int(u),
                                     coverage=float(covered[units == u].mean())))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--cells", default="docs/tables/mcd80_cells_seeded.csv")
    ap.add_argument("--out", default="docs/figures/ijphm_v2")
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(root / args.cells)
    r_cqr = (d.cqr_nom - d.const_nom) / d.const_nom * 100
    r_mcd = (d.mcd_nom - d.const_mc_nom) / d.const_mc_nom * 100

    # ---------------- fig 1 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, v, t in [(axes[0], r_cqr, "CQR vs constant width"),
                     (axes[1], r_mcd, "dropout scale vs same-centre constant")]:
        ax.hist(v, bins=np.arange(-30, 61, 3), color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.axvline(v.mean(), color="#C44E52", lw=1.0)
        ax.set_title(t, fontsize=8)
        ax.set_xlabel("relative Winkler change [%] (negative = better)", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("cells", fontsize=7)
    fig.suptitle("80 cells (4 subsets x 5 folds x 4 seeds), nominal level", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_effect_distribution.pdf")
    fig.savefig(out / "fig1_effect_distribution.png", dpi=200)
    plt.close(fig)

    # ---------------- fig 2 ----------------
    d2 = d.assign(r_mcd=r_mcd)
    fold = d2.groupby(["subset", "fold"]).r_mcd.mean().reset_index()
    fold["grp"] = np.where(fold.subset.isin(MULTI), "multi-condition", "single-condition")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for i, g in enumerate(["multi-condition", "single-condition"]):
        v = fold[fold.grp == g].r_mcd.to_numpy()
        x = np.full(v.size, i) + np.linspace(-0.08, 0.08, v.size)
        ax.scatter(x, v, s=14, color="#4C72B0" if i == 0 else "#C44E52", zorder=3)
        ax.hlines(v.mean(), i - 0.22, i + 0.22, color="k", lw=1.2, zorder=4)
        ax.annotate(f"mean {v.mean():+.1f}%", (i + 0.24, v.mean()), fontsize=6, va="center")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FD002+FD004\n(10 folds)", "FD001+FD003\n(10 folds)"], fontsize=7)
    ax.set_ylabel("fold-level relative Winkler change [%]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("dropout scale vs same-centre constant", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_fold_separation.pdf")
    fig.savefig(out / "fig2_fold_separation.png", dpi=200)
    plt.close(fig)

    # ---------------- fig 3 ----------------
    pe = cqr_per_engine(root)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    data = [pe[pe.subset == s].coverage.to_numpy() for s in SUBSETS]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.5),
                    medianprops=dict(color="k", lw=1.0))
    for patch in bp["boxes"]:
        patch.set_facecolor("#B0C4DE")
    ax.axhline(0.90, color="k", lw=0.8, ls="--")
    ax.set_xticklabels(SUBSETS, fontsize=7)
    ax.set_ylabel("per-engine coverage", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("frozen CQR, 80 cells", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_engine_coverage.pdf")
    fig.savefig(out / "fig3_engine_coverage.png", dpi=200)
    plt.close(fig)

    print(f"wrote 3 figures to {out}")
    print("per-engine coverage min by subset:",
          {s: round(float(pe[pe.subset == s].coverage.min()), 4) for s in SUBSETS})
    print("fold-level means:", fold.groupby("grp").r_mcd.mean().round(2).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


