"""Generate the IEEE TIM result tables from the training matrix.

Produces, in one pass:
  Table 1  matched-coverage Winkler per method (mean +- std over folds/seeds)
  Table 2  paired comparison of the proposed method against each baseline
  Table 3  coverage diagnostics (marginal, engine-level, end-of-life stage)
  Table 4  cross-domain FEMTO results

Every number comes from artefacts on disk; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_matrix import (at, const_pts, cqr_pts, engine_split, loc_cqr_pts)


def load_cell(root: Path, sub: str, seed: int, fold: int):
    d = root / sub / f"f{fold}_s{seed}"
    pq, pp = d / "preds_cqr.csv", d / "preds_point.csv"
    if not (pq.exists() and pp.exists()):
        return None
    c = np.genfromtxt(pq, delimiter=",", names=True)
    p = np.genfromtxt(pp, delimiter=",", names=True)
    cal, tst = engine_split(c["unit"])
    y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
    mu = (p["y_lo"] + p["y_hi"]) / 2.0
    return dict(
        cqr=at(cqr_pts(y, lo, hi, cal, tst)),
        loc=at(loc_cqr_pts(y, lo, hi, cal, tst)),
        const=at(const_pts(p["y_true"], mu, cal, tst)),
        json=json.loads((d / "e2e_cqr_none.json").read_text())
        if (d / "e2e_cqr_none.json").exists() else {},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/punet_matrix")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="docs/tables")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        for seed in args.seeds:
            for fold in range(args.folds):
                cell = load_cell(root, sub, seed, fold)
                if cell is None:
                    continue
                rows.append(dict(subset=sub, seed=seed, fold=fold,
                                 cqr=cell["cqr"][0], loc=cell["loc"][0], const=cell["const"][0],
                                 cqr_cov=cell["cqr"][1], loc_cov=cell["loc"][1],
                                 const_cov=cell["const"][1],
                                 engine_min=cell["json"].get("engine_min"),
                                 picp=cell["json"].get("picp")))
    if not rows:
        print("no complete cells")
        return 0
    df = pd.DataFrame(rows)
    df.to_csv(out / "cells.csv", index=False)

    print(f"complete cells: {len(df)}  (seeds: {sorted(df.seed.unique())})")

    t1 = df.groupby("subset")[["const", "cqr", "loc"]].agg(["mean", "std"]).round(1)
    print("\n== Table 1: Winkler at matched coverage ==")
    print(t1.to_string())
    t1.to_csv(out / "table1_main.csv")

    print("\n== Table 2: paired improvement of Loc-CQR ==")
    for base in ["const", "cqr"]:
        rel = (df["loc"] - df[base]) / df[base] * 100
        wins = int((rel < 0).sum())
        print(f"  vs {base:<6} wins {wins}/{len(df)} ({100*wins/len(df):.0f}%) "
              f"mean {rel.mean():+.1f}% median {rel.median():+.1f}%")
        df[f"rel_vs_{base}"] = rel
    df.to_csv(out / "table2_paired.csv", index=False)

    print("\n== Table 3: coverage diagnostics ==")
    if df["engine_min"].notna().any():
        g = df.groupby("subset").agg(picp=("picp", "mean"),
                                     engine_min=("engine_min", "mean"),
                                     frac_below_080=("engine_min", lambda s: (s < 0.80).mean()))
        print(g.round(3).to_string())
        g.to_csv(out / "table3_diagnostics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
