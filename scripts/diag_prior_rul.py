"""Diagnostic: RUL implied by the lifetime prior alone.

RUL_prior = clip(mu_T - t, 0, 125) on the same outer-val windows as M0.
Context for the Gate 0 report: how much RUL signal the causal prior carries
by itself, vs HGB direct RUL (H1) and the PE backbone (M0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(subset: str, base: Path) -> dict:
    rows = []
    for d in sorted((base / subset).glob("M0_f*_s*")):
        fold = int(d.name.split("_f")[1].split("_")[0])
        p = d / "val_predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        rul_prior = np.clip(df["mu_T"] - df["end_cycle"], 0, 125)
        rows.append(dict(
            fold=fold,
            rmse_pe=float(np.sqrt(np.mean((df["y_pred"] - df["y_true"]) ** 2))),
            rmse_prior=float(np.sqrt(np.mean((rul_prior - df["y_true"]) ** 2))),
            n_windows=len(df), n_units=int(df["unit"].nunique())))
    out = {"subset": subset, "folds": rows}
    if rows:
        # pooled (windows are engine-disjoint across folds)
        out["mean_prior_rmse"] = float(np.mean([r["rmse_prior"] for r in rows]))
        out["mean_pe_rmse"] = float(np.mean([r["rmse_pe"] for r in rows]))
    return out


def main():
    base = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "experiments/ida/gate0")
    report = {s: run(s, base) for s in ("FD001", "FD002")}
    out = base / "diag_prior_rul.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for s, r in report.items():
        if r["folds"]:
            print(f"{s}: prior-RUL RMSE {r['mean_prior_rmse']:.3f} vs "
                  f"PE {r['mean_pe_rmse']:.3f} "
                  f"(folds: {[round(f['rmse_prior'], 2) for f in r['folds']]})")
    print("saved", out)


if __name__ == "__main__":
    main()
