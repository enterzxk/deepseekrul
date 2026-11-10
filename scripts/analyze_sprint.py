"""Aggregate the 10h-sprint runs and compute paired arm comparisons.

Metric choice (important): the internal validation engines include windows at
RUL ~ 0 (each engine's true end), so "last window per engine" is trivially easy
and not comparable across arms. Primary metric here is ENGINE-MACRO RMSE:
compute MSE within each engine, then aggregate across engines. Secondary:
RMSE at a fixed operating point (the window whose y_true is closest to 60),
which is comparable across arms.

Recomputed from the saved preds.csv, so no retraining is needed.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_metrics(pred_path: Path) -> dict:
    d = pd.read_csv(pred_path)
    if d.empty:
        return {}
    e = d["y_hat"].to_numpy() - d["y_true"].to_numpy()
    d = d.assign(e2=e ** 2, ae=np.abs(e))
    g = d.groupby("unit").agg(mse=("e2", "mean"), mae=("ae", "mean"), n=("e2", "size"))
    out = dict(rmse_macro=float(np.sqrt(g["mse"].mean())),
               mae_macro=float(g["mae"].mean()),
               n_engines=int(len(g)))
    # fixed operating point: window closest to y_true = 60 per engine
    idx = (d["y_true"] - 60).abs().groupby(d["unit"]).idxmin()
    at60 = d.loc[idx]
    out["rmse_at60"] = float(np.sqrt(np.mean((at60["y_hat"] - at60["y_true"]) ** 2)))
    # last window per engine (secondary, trivially easy)
    last = d.sort_values("cutoff").groupby("unit").tail(1)
    out["rmse_last"] = float(np.sqrt(np.mean((last["y_hat"] - last["y_true"]) ** 2)))
    return out


def load() -> pd.DataFrame:
    rows = []
    for mf in glob.glob("experiments/tim_sprint/*/*/metrics.json"):
        d = json.loads(Path(mf).read_text(encoding="utf-8"))
        pf = Path(mf).parent / "preds.csv"
        if pf.exists():
            d.update(run_metrics(pf))
        d["dir"] = str(Path(mf).parent)
        rows.append(d)
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, base: str, cand: str, metric: str) -> dict:
    b = df[df.arm == base][["subset", "fold", "seed", metric]].dropna()
    c = df[df.arm == cand][["subset", "fold", "seed", metric]].dropna()
    m = b.merge(c, on=["subset", "fold", "seed"], suffixes=("_b", "_c"))
    if m.empty:
        return {}
    m["rel"] = (m[f"{metric}_c"] - m[f"{metric}_b"]) / m[f"{metric}_b"] * 100
    fold = m.groupby(["subset", "fold"])["rel"].mean()
    return dict(base=base, cand=cand, n_pairs=len(m),
                mean=float(m["rel"].mean()), median=float(m["rel"].median()),
                wins=int((m["rel"] < 0).sum()), folds_better=int((fold < 0).sum()), n_folds=int(len(fold)))


def main() -> int:
    df = load()
    if df.empty:
        print("no runs yet")
        return 0
    print(f"runs: {len(df)}")
    show = [c for c in ["subset", "arm", "seed", "fold", "rmse_macro", "mae_macro",
                        "rmse_at60", "rmse_last", "pooled_rmse", "best_epoch"] if c in df.columns]
    print(df[show].sort_values(["subset", "arm", "seed"]).round(3).to_string(index=False))

    print("\n--- per (subset, arm) mean over seeds ---")
    agg = [c for c in ["rmse_macro", "rmse_at60", "rmse_last"] if c in df.columns]
    print(df.groupby(["subset", "arm"])[agg].agg(["count", "mean"]).round(3).to_string())

    for metric in ["rmse_macro", "rmse_at60"]:
        if metric not in df.columns:
            continue
        print(f"\n--- paired comparisons on {metric} (negative = candidate better) ---")
        for base, cand in [("R", "A2"), ("A2-NE", "A2"), ("A2-NR", "A2"),
                           ("M0", "M1"), ("M2", "M1"), ("B2-G", "B2")]:
            r = paired(df, base, cand, metric)
            if r:
                print(f"{r['cand']:6s} vs {r['base']:6s}: pairs={r['n_pairs']:2d} "
                      f"mean={r['mean']:+.2f}% median={r['median']:+.2f}% "
                      f"wins={r['wins']}/{r['n_pairs']} folds_better={r['folds_better']}/{r['n_folds']}")

    mech = [c for c in ["switch_ratio", "spearman_dT_rul", "brier", "base_brier"] if c in df.columns]
    if mech:
        print("\n--- mechanism diagnostics (per subset/arm mean) ---")
        print(df.groupby(["subset", "arm"])[mech].mean().round(3).to_string())

    df.to_csv("docs/overnight_tim_research/sprint_runs.csv", index=False)
    print("\nsaved docs/overnight_tim_research/sprint_runs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
