"""Independent constant-scale block conformal baseline for RUL intervals.

Deliberately standalone: it does NOT import scripts/dsh_conformal.py, so the
existing exploration script stays untouched. Only numpy / pandas are used.

Input: a predictions CSV with columns unit_id, cutoff, y_true, y_pred
       (or the matched_cutoff_predictions.csv layout produced by the pipeline).
Output: results/punet/baseline_constcp/<subset>.json

Usage:
    python scripts/run_baseline_constcp.py --csv path/to/preds.csv --subset FD001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.punet.conformal import BlockConformalCalibrator, block_split  # noqa: E402
from src.punet.diagnostics import engine_bootstrap_ci, stratified_coverage  # noqa: E402
from src.punet.metrics import engine_coverage, picp, pinaw, winkler  # noqa: E402

COLUMN_ALIASES = {
    "unit_id": ["unit_id", "unit", "engine", "engine_id"],
    "cutoff": ["cutoff", "c", "cycle", "t"],
    "y_true": ["y_true", "target", "rul_true", "y"],
    "y_pred": ["y_pred", "prediction", "pred", "y_hat"],
}


def _pick(df: pd.DataFrame, key: str) -> str:
    for name in COLUMN_ALIASES[key]:
        if name in df.columns:
            return name
    raise KeyError(f"none of {COLUMN_ALIASES[key]} found for {key}; columns={list(df.columns)}")


def load_predictions(path: Path) -> pd.DataFrame:
    """Load a predictions CSV.

    Some pipeline artefacts (matched_cutoff_predictions.csv) store one row per
    engine with no cutoff column. In that case a synthetic within-engine cutoff
    index is added so the block splitter still has a temporal ordering; the
    split then degenerates to an engine-level split, which is the intended
    semantics for engine-matched predictions.
    """
    df = pd.read_csv(path)
    ren = {_pick(df, k): k for k in COLUMN_ALIASES if any(a in df.columns for a in COLUMN_ALIASES[k])}
    df = df.rename(columns=ren)
    if "cutoff" not in df.columns:
        df = df.sort_values(["unit_id"]).reset_index(drop=True)
        df["cutoff"] = df.groupby("unit_id").cumcount()
    df = df[["unit_id", "cutoff", "y_true", "y_pred"]].dropna()
    return df.sort_values(["unit_id", "cutoff"]).reset_index(drop=True)


def run(df: pd.DataFrame, delta: float, seed: int) -> dict:
    units = df["unit_id"].to_numpy()
    cutoffs = df["cutoff"].to_numpy()
    y = df["y_true"].to_numpy(float)
    y_hat = df["y_pred"].to_numpy(float)
    w = np.ones_like(y)  # constant scale: this baseline has no prior information

    split = block_split(units, cutoffs, seed=seed)
    scores = BlockConformalCalibrator.scores(y[split.calib], y_hat[split.calib], w[split.calib])
    cal = BlockConformalCalibrator(delta=delta).fit(scores)
    lo, hi = cal.interval(y_hat[split.test], w[split.test])

    y_t = y[split.test]
    covered = ((y_t >= lo) & (y_t <= hi)).astype(float)
    cov = engine_coverage(units[split.test], y_t, lo, hi)
    ci = engine_bootstrap_ci(units[split.test], covered, n_boot=5000, seed=seed)

    return {
        "method": "constant_scale_block_cp",
        "n_total": int(len(df)),
        "split": split.describe(),
        "q_hat": cal.q_hat,
        "picp": picp(y_t, lo, hi),
        "pinaw": pinaw(y_t, lo, hi),
        "winkler": winkler(y_t, lo, hi, alpha=delta),
        "engine_min": cov["min"],
        "engine_p05": cov["p05"],
        "engine_median": cov["median"],
        "bootstrap_ci": list(ci),
        "rmse": float(np.sqrt(np.mean((y_t - y_hat[split.test]) ** 2))),
        "diagnostics": stratified_coverage(
            units[split.test], cutoffs[split.test], y[split.test], y_t, lo, hi
        ),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--subset", default="FD001")
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(ROOT / "results" / "punet" / "baseline_constcp"))
    args = p.parse_args(argv)

    df = load_predictions(Path(args.csv))
    result = run(df, args.delta, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.subset}.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}, ensure_ascii=False))
    print("written", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
