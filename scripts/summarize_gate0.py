"""Gate 0 verdict: engine-clustered paired bootstrap for IDA-PE arms.

Reads experiments/ida/gate0/<subset>/<arm>_f<fold>_s<seed>/val_predictions.csv
(unit, end_cycle, y_true, y_pred, mu_T, sigma_T), aggregates per-engine MSE,
and bootstraps engines (never windows). Emits the frozen criteria G0.1-G0.4
from gates.json with a PASS/FAIL verdict line.

DeltaRMSE(a - b) = RMSE(a) - RMSE(b) computed on the same engine resample.
Each engine is a validation engine in exactly one outer fold, so pooling folds
concatenates independent engine-level statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPS = 5000


def engine_mse(frame: pd.DataFrame) -> pd.Series:
    """Per-engine mean squared error over that engine's windows."""
    d = (frame["y_pred"] - frame["y_true"]) ** 2
    return d.groupby(frame["unit"]).mean()


def load_arm(base, subset, arm, fold, seed):
    p = base / subset / f"{arm}_f{fold}_s{seed}" / "val_predictions.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def collect(base, subset, arm_a, arm_b, seed):
    """Per-fold paired engine stats + fold-level bootstrap."""
    rows = []
    for fold in range(5):
        fa = load_arm(base, subset, arm_a, fold, seed)
        fb = load_arm(base, subset, arm_b, fold, seed)
        if fa is None or fb is None:
            continue
        ma, mb = engine_mse(fa), engine_mse(fb)
        units = ma.index.intersection(mb.index)
        da, db = ma.loc[units].to_numpy(), mb.loc[units].to_numpy()
        rng = np.random.default_rng(7 + fold)
        idx = rng.integers(0, len(units), size=(REPS, len(units)))
        d = np.sqrt(da[idx].mean(axis=1)) - np.sqrt(db[idx].mean(axis=1))
        rows.append(dict(
            fold=fold, n_units=int(len(units)),
            rmse_a=float(np.sqrt(da.mean())), rmse_b=float(np.sqrt(db.mean())),
            point=float(d.mean()), ci=[float(x) for x in np.percentile(d, [2.5, 97.5])],
            _ma=da, _mb=db))
    return rows


def pooled_delta(rows, bseed=99):
    """Pool folds: engines are independent across folds; one pooled bootstrap."""
    all_a = np.concatenate([r["_ma"] for r in rows])
    all_b = np.concatenate([r["_mb"] for r in rows])
    rng = np.random.default_rng(bseed)
    idx = rng.integers(0, len(all_a), size=(REPS, len(all_a)))
    dd = np.sqrt(all_a[idx].mean(axis=1)) - np.sqrt(all_b[idx].mean(axis=1))
    lo, hi = np.percentile(dd, [2.5, 97.5])
    return dict(n_engines=int(len(all_a)),
                rmse_a=float(np.sqrt(all_a.mean())),
                rmse_b=float(np.sqrt(all_b.mean())),
                point=float(dd.mean()), ci=[float(lo), float(hi)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="experiments/ida/gate0")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default="experiments/ida/gate0/gate0_verdict.json")
    args = ap.parse_args()
    base = ROOT / args.base

    report = {"seed": args.seed, "pairs": {}}
    verdicts = []
    comparisons = [
        ("FD002", "M1", "M0", "G0.1", "upper<0"),
        ("FD002", "M1_shuf", "M1", "G0.2", "lower>0"),
        ("FD002", "M1_const", "M1", "G0.3", "lower>0"),
        ("FD001", "M1", "M0", "G0.4", "upper<+0.5"),
    ]
    for subset, arm_a, arm_b, gid, rule in comparisons:
        rows = collect(base, subset, arm_a, arm_b, args.seed)
        if len(rows) < 5:
            report["pairs"][gid] = dict(status="INCOMPLETE", n_folds=len(rows),
                                        subset=subset, pair=f"{arm_a}-{arm_b}")
            verdicts.append((gid, "INCOMPLETE"))
            print(f"{gid} [{subset} {arm_a} vs {arm_b}] only {len(rows)}/5 folds -> INCOMPLETE")
            continue
        pooled = pooled_delta(rows)
        lo, hi = pooled["ci"]
        ok = {"upper<0": hi < 0, "lower>0": lo > 0,
              "upper<+0.5": hi < 0.5}[rule]
        status = "PASS" if ok else "FAIL"
        verdicts.append((gid, status))
        report["pairs"][gid] = dict(
            status=status, subset=subset, pair=f"{arm_a} - {arm_b}", rule=rule,
            rmse_a=pooled["rmse_a"], rmse_b=pooled["rmse_b"],
            delta_rmse=pooled["point"], ci95=pooled["ci"],
            n_engines=pooled["n_engines"],
            per_fold=[{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in rows])
        print(f"{gid} [{subset} {arm_a} vs {arm_b}] "
              f"RMSE {pooled['rmse_a']:.3f} vs {pooled['rmse_b']:.3f}  "
              f"dRMSE {pooled['point']:+.3f} CI [{lo:+.3f}, {hi:+.3f}]  -> {status}")

    all_pass = all(v == "PASS" for _, v in verdicts)
    report["verdict"] = ("GO_TO_GATE1" if all_pass else
                         ("INCOMPLETE" if any(v == "INCOMPLETE" for _, v in verdicts)
                          else "STOP"))
    report["criteria"] = dict(verdicts)
    print("=" * 60)
    print("GATE 0 VERDICT:", report["verdict"])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["verdict"] != "INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
