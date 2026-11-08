"""Aggregate coverage diagnostics across the training matrix.

Goal item (5): full engine-level and stage-wise coverage diagnostics. Reads the
per-run JSONs produced by run_e2e_cqr.py and summarises:

  * engine-level minimum coverage (per unit, worst case)
  * stage-wise coverage (by degradation clock tau)
  * how often the nominal 0.90 is missed, and by how much
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/punet_matrix")
    ap.add_argument("--pattern", default="e2e_cqr_none.json")
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for p in sorted(root.glob(f"*/*/{args.pattern}")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        diag = d.get("diagnostics", {})
        bins = diag.get("stage_bins", [])
        rec = {
            "subset": d.get("subset"),
            "fold": d.get("fold"),
            "picp": d.get("picp"),
            "engine_min": d.get("engine_min"),
            "engine_p05": d.get("engine_p05"),
            "engine_median": d.get("engine_median"),
        }
        for i, b in enumerate(bins):
            rec[f"stage{i}_cov"] = b["coverage"]
        rows.append(rec)

    if not rows:
        print("no runs found")
        return 0
    df = pd.DataFrame(rows)
    print("runs:", len(df))
    print("\n== engine-level coverage ==")
    print(df.groupby("subset")[["picp", "engine_min", "engine_p05", "engine_median"]]
            .agg(["mean", "std"]).round(3).to_string())
    print("\n== how often engine_min < 0.80 ==")
    print(df.assign(bad=df["engine_min"] < 0.80).groupby("subset")["bad"].mean().round(2).to_string())
    stage_cols = [c for c in df.columns if c.startswith("stage") and c.endswith("_cov")]
    if stage_cols:
        print("\n== stage-wise coverage (mean over runs) ==")
        print(df.groupby("subset")[stage_cols].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
