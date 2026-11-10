"""Aggregate the corrected bearing LOBO results (v2).

Reports per-bearing error at the fixed life stages, macro means, pairwise win
rates, bearing-level bootstrap intervals (the bearing is the resampling unit),
and a by-operating-condition breakdown (Bearing1_x / 2_x / 3_x).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/bearing_tim2"
ARMS = ["A", "C", "C-Fixed", "C-MLP"]


def load() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(DIR / "*" / "*.csv")):
        d = pd.read_csv(f)
        d["arm"] = Path(f).parent.name
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def per_bearing(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["arm", "holdout"]).apply(
        lambda x: pd.Series(dict(rmse=float(np.sqrt(np.mean(x.err ** 2))),
                                 mae=float(np.mean(np.abs(x.err))),
                                 n=len(x))), include_groups=False).reset_index()
    return g


def main() -> int:
    df = load()
    pb = per_bearing(df)
    piv = pb.pivot(index="holdout", columns="arm", values="rmse")
    print("--- per-bearing RMSE at fixed life stages (50/70/85/95%) ---")
    print(piv.round(1).to_string())
    print("\n--- macro mean ---")
    print(piv.mean().round(1).to_string())

    print("\n--- pairwise relative change (negative = candidate better) ---")
    for base, cand in [("A", "C"), ("C-Fixed", "C"), ("C-MLP", "C"), ("A", "C-Fixed")]:
        if base in piv and cand in piv:
            d = (piv[cand] - piv[base]) / piv[base] * 100
            print(f"  {cand:7s} vs {base:7s}: mean={d.mean():+.1f}% median={d.median():+.1f}% "
                  f"wins={(d<0).sum()}/{len(d)}")

    print("\n--- bearing-level bootstrap (resample bearings, 5000 reps) ---")
    rng = np.random.default_rng(0)
    for base, cand in [("A", "C"), ("C-Fixed", "C"), ("C-MLP", "C")]:
        if base not in piv or cand not in piv:
            continue
        b = piv[base].to_numpy(); c = piv[cand].to_numpy()
        n = len(b); rel = []
        for _ in range(5000):
            idx = rng.integers(0, n, n)
            rel.append((np.sqrt(np.mean(c[idx] ** 2)) / np.sqrt(np.mean(b[idx] ** 2)) - 1) * 100)
        rel = np.array(rel)
        print(f"  {cand:7s} vs {base:7s}: {rel.mean():+.1f}%  95%CI=[{np.percentile(rel,2.5):+.1f}, "
              f"{np.percentile(rel,97.5):+.1f}]  P(rel<0)={float((rel<0).mean()):.3f}")

    print("\n--- by operating condition (bearing family) ---")
    piv2 = piv.copy()
    piv2["cond"] = [i.split("_")[1][:7] for i in piv2.index]
    print(piv2.groupby("cond").mean().round(1).to_string())

    print("\n--- per-stage error (all bearings, mean absolute error) ---")
    df["stage"] = df.groupby(["arm", "holdout"]).cumcount()
    print(df.groupby(["arm", "stage"]).err.apply(lambda e: float(np.mean(np.abs(e)))).unstack().round(1).to_string())

    pb.to_csv(ROOT / "docs/bearing_tim/per_bearing_v2.csv", index=False)
    print("\nsaved docs/bearing_tim/per_bearing_v2.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
