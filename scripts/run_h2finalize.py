"""H2 finalize: choose the PE/HGB mixing weight on inner CV, then build the
H2 outer-val predictions.

w*(subset, fold) = argmin over the grid of pooled inner-val RMSE of
w·PE + (1−w)·HGB (from run_h2inner outputs). Outer-val H2 predictions =
w*·M0(seed 101) + (1−w*)·H1, written in the standard val_predictions.csv
format so summarize_gate0/summarize_gate1 consume it unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def pick_weight(inner_dir, subset, fold):
    frames = []
    for k in range(4):
        p = inner_dir / subset / f"f{fold}_k{k}.csv"
        if not p.exists():
            return None, None
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    y = df["y_true"].to_numpy()
    pe = df["pe_pred"].to_numpy()
    hg = df["hgb_pred"].to_numpy()
    rmses = []
    for w in GRID:
        rmses.append(float(np.sqrt(np.mean((w * pe + (1 - w) * hg - y) ** 2))))
    i = int(np.argmin(rmses))
    return float(GRID[i]), rmses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-dir", default="experiments/ida/gate1/h2inner")
    ap.add_argument("--gate0-dir", default="experiments/ida/gate0")
    ap.add_argument("--out", default="experiments/ida/gate1")
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()
    inner_dir, gate0_dir, out = ROOT / args.inner_dir, ROOT / args.gate0_dir, ROOT / args.out

    report = {}
    for subset in ("FD001", "FD002"):
        weights = {}
        for fold in range(5):
            w, _ = pick_weight(inner_dir, subset, fold)
            if w is None:
                continue
            m0 = gate0_dir / subset / f"M0_f{fold}_s{args.seed}" / "val_predictions.csv"
            h1 = gate0_dir / subset / f"H1_f{fold}_s0" / "val_predictions.csv"
            if not (m0.exists() and h1.exists()):
                continue
            a = pd.read_csv(m0)
            b = pd.read_csv(h1)[["unit", "end_cycle", "y_pred"]]
            merged = a.merge(b, on=["unit", "end_cycle"], suffixes=("", "_h1"))
            merged["y_pred"] = w * merged["y_pred"] + (1 - w) * merged["y_pred_h1"]
            d = out / subset / f"H2_f{fold}_s{args.seed}"
            d.mkdir(parents=True, exist_ok=True)
            merged[["unit", "end_cycle", "y_true", "y_pred"]].to_csv(
                d / "val_predictions.csv", index=False)
            weights[fold] = w
        report[subset] = weights
        print(subset, "weights:", weights)
    (out / "h2_weights.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
