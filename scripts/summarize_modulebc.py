"""Module B/C verdict vs M1p and condition-isolation controls (FD002).

Arms: MB_tau/MB_abs/MB_shuf (Module B), MC_tau/MC_abs/MC_shuf/MC_static
(Module C), all trained with context injection fixed to M1p so that ONLY
the module-level condition signal differs across arms.

Keep rule (from the frozen gating spec, "Module B/C: one zero-init small
experiment each, vs M1p + shuffled-tau control"): the tau arm must beat M1p
with a 95% engine-clustered bootstrap CI excluding 0 AND beat both the abs
(absolute-time) and shuf (shuffled-tau) controls the same way. Otherwise the
module is dropped; the paper mainline (M1p) is unaffected either way.

Paths: module arms -> experiments/ida/moduleBC/FD002/{arm}_f{fold}_s101
       M1p ref    -> experiments/ida/gate0/FD002/M1p_f{fold}_s101
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
MBC = ROOT / "experiments" / "ida" / "moduleBC"


def arm_dir(arm: str, fold: int, seed: int) -> Path:
    if arm in ("M0", "M1", "M1p", "M1_shuf", "M1_const"):
        return GATE0 / "FD002" / f"{arm}_f{fold}_s{seed}"
    return MBC / "FD002" / f"{arm}_f{fold}_s{seed}"


def engine_mse(frame: pd.DataFrame) -> pd.Series:
    d = (frame["y_pred"] - frame["y_true"]) ** 2
    return d.groupby(frame["unit"]).mean()


def pooled_delta(arm_a: str, arm_b: str, seed: int):
    """Pool folds; engine-clustered bootstrap, delta = RMSE(a) - RMSE(b)."""
    ma_all, mb_all, per_fold = [], [], []
    for fold in range(5):
        pa = arm_dir(arm_a, fold, seed) / "val_predictions.csv"
        pb = arm_dir(arm_b, fold, seed) / "val_predictions.csv"
        if not pa.exists() or not pb.exists():
            return None
        ma = engine_mse(pd.read_csv(pa))
        mb = engine_mse(pd.read_csv(pb))
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


def judge_module(prefix: str, seed: int, report: dict) -> str:
    """Run the three frozen comparisons for one module; return keep/drop."""
    tau = f"{prefix}_tau"
    pairs = [(f"{prefix}.1", tau, "M1p"),
             (f"{prefix}.2", tau, f"{prefix}_abs"),
             (f"{prefix}.3", tau, f"{prefix}_shuf")]
    results = {}
    all_pass = True
    for pid, a, b in pairs:
        r = pooled_delta(a, b, seed)
        if r is None:
            results[pid] = {"status": "INCOMPLETE", "pair": f"{a} - {b}"}
            all_pass = False
            print(f"{pid} [{a} vs {b}] INCOMPLETE")
            continue
        lo, hi = r["ci95"]
        ok = hi < 0
        all_pass &= ok
        results[pid] = dict(status="PASS" if ok else "FAIL",
                            pair=f"{a} - {b}", **r)
        print(f"{pid} [{a} vs {b}] RMSE {r['rmse_a']:.3f} vs "
              f"{r['rmse_b']:.3f}  dRMSE {r['delta_rmse']:+.3f} "
              f"CI [{lo:+.3f}, {hi:+.3f}]  -> {'PASS' if ok else 'FAIL'}")
    verdict = ("KEEP" if all_pass and all(
        v.get("status") == "PASS" for v in results.values())
               else ("DROP" if all(v.get("status") != "INCOMPLETE"
                                   for v in results.values()) else "INCOMPLETE"))
    report["modules"][prefix] = {"comparisons": results, "verdict": verdict}
    print(f"MODULE {prefix} VERDICT: {verdict}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default="experiments/ida/moduleBC/modulebc_verdict.json")
    args = ap.parse_args()

    report = {"seed": args.seed, "modules": {}}
    print(f"===== Module B (seed {args.seed}) =====")
    vb = judge_module("MB", args.seed, report)
    print(f"===== Module C (seed {args.seed}) =====")
    vc = judge_module("MC", args.seed, report)

    complete = {vb, vc}.issubset({"KEEP", "DROP"})
    final = ({vb, vc} - {"DROP"})
    report["final_verdict"] = ("INCOMPLETE" if not complete else
                               ("KEEP_" + "_".join(sorted(final))
                                if final else "DROP_BOTH"))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 60)
    print("MODULE B/C FINAL VERDICT:", report["final_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
