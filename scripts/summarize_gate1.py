"""Gate 1 verdict: M1/M1p vs H1/H2/H3 on FD002 (frozen criteria G1.1-G1.3).

Reads outer-val predictions from their canonical locations:
  M0/M1/M1p -> experiments/ida/gate0/FD002/{arm}_f{fold}_s{seed}
  H1        -> experiments/ida/gate0/FD002/H1_f{fold}_s0   (CPU arm, seed 0)
  H2/H3     -> experiments/ida/gate1/FD002/{arm}_f{fold}_s101
Statistics are identical to summarize_gate0 (engine-clustered pooled
bootstrap, 5000 reps); this script only changes the arm paths and the
verdict pairs. Criteria come from gates.json gate1 (G1.1/G1.2/G1.3):
DeltaRMSE(M1 or M1p vs H) CI must exclude 0 with the PE arm better.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPS = 5000
GATE0 = ROOT / "experiments" / "ida" / "gate0"
GATE1 = ROOT / "experiments" / "ida" / "gate1"


def arm_dir(arm: str, fold: int, seed: int) -> Path:
    if arm in ("M0", "M1", "M1p", "M1_shuf", "M1_const"):
        return GATE0 / "FD002" / f"{arm}_f{fold}_s{seed}"
    if arm == "H1":
        return GATE0 / "FD002" / f"H1_f{fold}_s0"
    return GATE1 / "FD002" / f"{arm}_f{fold}_s{seed}"


def engine_mse(frame: pd.DataFrame) -> pd.Series:
    d = (frame["y_pred"] - frame["y_true"]) ** 2
    return d.groupby(frame["unit"]).mean()


def pooled_delta(arm_a: str, arm_b: str, seed: int):
    """Pool folds: engines are independent across folds; one bootstrap."""
    ma_all, mb_all, per_fold = [], [], []
    for fold in range(5):
        pa = arm_dir(arm_a, fold, seed) / "val_predictions.csv"
        pb = arm_dir(arm_b, fold, seed) / "val_predictions.csv"
        if not pa.exists() or not pb.exists():
            return None
        fa, fb = pd.read_csv(pa), pd.read_csv(pb)
        ma, mb = engine_mse(fa), engine_mse(fb)
        units = ma.index.intersection(mb.index)
        ma_all.append(ma.loc[units].to_numpy())
        mb_all.append(mb.loc[units].to_numpy())
        per_fold.append(dict(fold=fold, n_units=int(len(units)),
                             rmse_a=float(np.sqrt(ma_all[-1].mean())),
                             rmse_b=float(np.sqrt(mb_all[-1].mean()))))
    all_a = np.concatenate(ma_all)
    all_b = np.concatenate(mb_all)
    rng = np.random.default_rng(99)
    idx = rng.integers(0, len(all_a), size=(REPS, len(all_a)))
    dd = np.sqrt(all_a[idx].mean(axis=1)) - np.sqrt(all_b[idx].mean(axis=1))
    lo, hi = np.percentile(dd, [2.5, 97.5])
    return dict(n_engines=int(len(all_a)),
                rmse_a=float(np.sqrt(all_a.mean())),
                rmse_b=float(np.sqrt(all_b.mean())),
                delta_rmse=float(dd.mean()), ci95=[float(lo), float(hi)],
                per_fold=per_fold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default="experiments/ida/gate1/gate1_verdict.json")
    args = ap.parse_args()

    report = {"seed": args.seed, "pairs": {}, "criteria": {}}
    comparisons = [
        ("G1.1", "M1", "H1"), ("G1.1", "M1p", "H1"),
        ("G1.2", "M1", "H2"), ("G1.2", "M1p", "H2"),
        ("G1.3", "M1", "H3"), ("G1.3", "M1p", "H3"),
    ]
    wins = {"M1": [], "M1p": []}
    for gid, a, b in comparisons:
        r = pooled_delta(a, b, args.seed)
        if r is None:
            report["pairs"][f"{gid}:{a}-{b}"] = {"status": "INCOMPLETE"}
            print(f"{gid} [{a} vs {b}] INCOMPLETE")
            continue
        lo, hi = r["ci95"]
        ok = hi < 0  # PE arm better, CI excludes 0
        status = "PASS" if ok else "FAIL"
        if ok:
            wins[a].append(b)
        report["pairs"][f"{gid}:{a}-{b}"] = dict(status=status, pair=f"{a} - {b}",
                                                 **r)
        print(f"{gid} [{a} vs {b}] RMSE {r['rmse_a']:.3f} vs {r['rmse_b']:.3f}  "
              f"dRMSE {r['delta_rmse']:+.3f} CI [{lo:+.3f}, {hi:+.3f}]  -> {status}")

    # decision rule: at least one of M1/M1p beats ALL of H1, H2, H3
    allh = {"H1", "H2", "H3"}
    winners = [a for a in ("M1", "M1p") if allh.issubset(set(wins[a]))]
    report["winners"] = {a: sorted(wins[a]) for a in ("M1", "M1p")}
    report["verdict"] = ("GO_TO_MODULE_BC" if winners else
                         ("HGB_SUFFICES_ANALYSIS_PAPER" if not any(
                             "INCOMPLETE" in str(v) for v in report["pairs"].values())
                          else "INCOMPLETE"))
    print("=" * 60)
    print("GATE 1 VERDICT:", report["verdict"], "| winners:", winners or "none")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
