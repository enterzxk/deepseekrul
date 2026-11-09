"""Monte-Carlo noise floor for the MC-Dropout baseline.

The first 80-cell pass was unseeded; the second fixes RNG = cell seed. Both
passes use the same checkpoints and the same analysis protocol, so the
difference between them is exactly the MC sampling error of T=30 dropout
passes -- the quantity that decides whether small effects (e.g. a median
difference of ~2%) are meaningful.

Prints, over all 80 cells:
  - |delta| of the primary per-cell metric r_scale_nom (mcd vs same-centre
    constant width, nominal level, %)
  - |delta| of realised coverage and of the aligned metric
  - how many cells change sign of "mcd better than const"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_mcd80 as A  # noqa: E402


def with_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["r_scale_nom"] = (df["mcd_nom"] - df["const_mc_nom"]) / df["const_mc_nom"] * 100
    df["r_scale_al"] = (df["mcd"] - df["const_mc"]) / df["const_mc"] * 100
    return df


def main() -> int:
    first = with_metrics(A.build_frame("experiments/punet_matrix", "experiments/punet_mcd80"))
    second = with_metrics(A.build_frame("experiments/punet_matrix", "experiments/punet_mcd80_seeded"))
    key = ["subset", "seed", "fold"]
    j = first.merge(second, on=key, suffixes=("_p1", "_p2"))
    j = j[j["mcd_nom_p1"].notna() & j["mcd_nom_p2"].notna()]
    print(f"cells compared: {len(j)}")

    for col, label in [("r_scale_nom", "primary metric r_scale_nom [%]"),
                       ("mcd_nom_cov", "realised coverage"),
                       ("mcd_nom", "nominal Winkler"),
                       ("r_scale_al", "aligned metric [%]")]:
        d = (j[f"{col}_p1"] - j[f"{col}_p2"]).abs().dropna()
        if not len(d):
            continue
        print(f"{label}: median |d|={d.median():.3f} p90={d.quantile(0.9):.3f} "
              f"max={d.max():.3f}")

    s1 = np.sign(j["r_scale_nom_p1"])
    s2 = np.sign(j["r_scale_nom_p2"])
    print(f"sign flips (mcd better/worse): {int((s1 != s2).sum())}/{len(j)}")
    print(f"pass1 mean={j['r_scale_nom_p1'].mean():+.2f}% median={j['r_scale_nom_p1'].median():+.2f}%")
    print(f"pass2 mean={j['r_scale_nom_p2'].mean():+.2f}% median={j['r_scale_nom_p2'].median():+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
