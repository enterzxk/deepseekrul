#!/usr/bin/env python3
"""EAC Phase A (G1 gate) summary.

Reads ONLY metrics.json + val_predictions.csv written by run_eac.py, plus the
raw train_FDxxx.txt for lifetime-stratified buckets. No numbers are hand
written anywhere (Integrity rule, spec section 9).

Usage:  python scripts/summarize_eac.py [results/eac/phaseA]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataset import load_cmapss_train  # noqa: E402

REPS = 4000
BOOT_SEED = 2026
VARIANTS = ("V0", "V1", "V6")
SUBSETS = ("FD001", "FD002")


def load_runs(root):
    runs = []
    for m in sorted(glob.glob(os.path.join(root, "*", "*", "metrics.json"))):
        d = json.load(open(m, encoding="utf-8"))
        d["_pred"] = os.path.join(os.path.dirname(m), "val_predictions.csv")
        d["_dir"] = os.path.dirname(m)
        runs.append(d)
    return runs


def load_preds(path):
    df = pd.read_csv(path)
    return df["unit_id"].to_numpy(np.int64), \
        df["y_true"].to_numpy(np.float64), \
        df["y_pred"].to_numpy(np.float64)


def rmse_table(runs):
    by = defaultdict(dict)
    for r in runs:
        by[(r["subset"], r["variant"])][r["seed"]] = r["best_val_rmse"]
    print(f"{'subset':7s} {'variant':8s} {'n_seeds':>7s} "
          f"{'RMSE mean':>10s} {'sd':>6s}   seeds")
    for (sub, var), d in sorted(by.items()):
        vals = list(d.values())
        print(f"{sub:7s} {var:8s} {len(vals):>7d} {np.mean(vals):>10.3f} "
              f"{np.std(vals):>6.3f}   "
              f"{' '.join(f'{v:.2f}' for v in sorted(vals))}")
    return by


def unit_mse(path):
    units, ys, preds = load_preds(path)
    sq = (preds - ys) ** 2
    return pd.DataFrame({"sq": sq, "u": units}).groupby("u").mean()["sq"]


def paired_bootstrap(runs, subset, va, vb, reps=REPS, seed=BOOT_SEED):
    """Paired engine-cluster bootstrap, seeds as independent replicates:
    the resampling unit is the (seed, unit) pair whose value is the unit's
    mean squared error under a minus under b (negative => a better).
    Negative d => a better than b."""
    per_seed = defaultdict(dict)
    for r in runs:
        if r["subset"] != subset or r["variant"] not in (va, vb):
            continue
        key = va if r["variant"] == va else vb
        per_seed[r["seed"]][key] = unit_mse(r["_pred"])
    seeds = sorted(s for s, d in per_seed.items() if len(d) == 2)
    if not seeds:
        return None
    dmat = []
    for s in seeds:
        a, b = per_seed[s][va].align(per_seed[s][vb], join="inner")
        dmat.append((a - b).to_numpy())
    d_all = np.concatenate(dmat)                   # (n_seeds * n_units,)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d_all), size=(reps, len(d_all)))
    # the gate is written on dRMSE units: resample (seed, unit) pairs and
    # take the RMSE difference of the pooled MSE in each replicate
    mse_split = {"a": [], "b": []}
    for s in seeds:
        a, b = per_seed[s][va].align(per_seed[s][vb], join="inner")
        mse_split["a"].append(a.to_numpy())
        mse_split["b"].append(b.to_numpy())
    mse_a = np.concatenate(mse_split["a"])
    mse_b = np.concatenate(mse_split["b"])
    d_rmse = (np.sqrt(mse_a[idx].mean(axis=1))
              - np.sqrt(mse_b[idx].mean(axis=1)))
    lo, hi = np.percentile(d_rmse, [2.5, 97.5])
    rmses = {v: [] for v in (va, vb)}
    for r in runs:
        if r["subset"] == subset and r["variant"] in rmses:
            rmses[r["variant"]].append(r["best_val_rmse"])
    return dict(n_units=len(d_all), n_seeds=len(seeds),
                rmse_a=float(np.mean(rmses[va])),
                rmse_b=float(np.mean(rmses[vb])),
                dmse_mean=float(d_all.mean()),
                dmse_ci=[float(lo), float(hi)])


def lifetime_buckets(subset, runs):
    """Train-lifetime tercile boundaries; per-run stratified RMSE/bias."""
    frame = load_cmapss_train(os.path.join(ROOT, "data", "CMAPSS"), subset)
    life = frame.groupby("unit").size()
    q1, q2 = life.quantile([1 / 3, 2 / 3])
    lab = pd.cut(life, [-np.inf, q1, q2, np.inf],
                 labels=["short", "mid", "long"])
    table = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["subset"] != subset:
            continue
        units, trues, preds = load_preds(r["_pred"])
        df = pd.DataFrame({"u": units, "err": preds - trues})
        df["bucket"] = df["u"].map(lab)
        g = df.groupby("bucket")["err"]
        for b, err in g:
            table[r["variant"]][str(b)].append(
                (float(np.sqrt((err ** 2).mean())), float(err.mean())))
    print(f"\n=== {subset} lifetime terciles "
          f"(q1={q1:.0f}, q2={q2:.0f} cycles; cell = RMSE/bias over seeds)")
    for var in VARIANTS:
        if var not in table:
            continue
        cells = []
        for b in ("short", "mid", "long"):
            v = table[var].get(b, [])
            if v:
                cells.append(f"{np.mean([x[0] for x in v]):6.2f}/"
                             f"{np.mean([x[1] for x in v]):+6.2f}")
            else:
                cells.append("  n/a  ")
        print(f"  {var:6s} " + " ".join(f"{c:>14s}" for c in cells))


def gate(by, runs):
    print("\n=== G1 gate (preregistered: dRMSE >= 0.3, CI excludes 0, "
          "V6 gain < half of V1 gain)")
    verdict = "GO"
    for sub in SUBSETS:
        d0 = by.get((sub, "V0"), {})
        d1 = by.get((sub, "V1"), {})
        d6 = by.get((sub, "V6"), {})
        if not (d0 and d1 and d6):
            print(f"  {sub}: incomplete variants, skip")
            verdict = "INCOMPLETE"
            continue
        gain_v1 = float(np.mean(list(d0.values())) - np.mean(list(d1.values())))
        gain_v6 = float(np.mean(list(d0.values())) - np.mean(list(d6.values())))
        r1 = paired_bootstrap(runs, sub, "V1", "V0")
        r6 = paired_bootstrap(runs, sub, "V6", "V0")
        c1 = f"dRMSE {gain_v1:+.3f} CI[{r1['dmse_ci'][0]:+.2f},{r1['dmse_ci'][1]:+.2f}]" if r1 else "n/a"
        c6 = f"dRMSE {gain_v6:+.3f} CI[{r6['dmse_ci'][0]:+.2f},{r6['dmse_ci'][1]:+.2f}]" if r6 else "n/a"
        ok_main = gain_v1 >= 0.3 and r1 and r1["dmse_ci"][1] < 0
        ok_ctrl = gain_v6 < 0.5 * gain_v1
        sub_verdict = "PASS" if (ok_main and ok_ctrl) else "FAIL"
        if sub_verdict == "FAIL":
            verdict = "NO-GO" if verdict != "INCOMPLETE" else verdict
        print(f"  {sub}: V1 vs V0 {c1} (n={r1['n_units'] if r1 else '-'}) | "
              f"V6 vs V0 {c6}")
        print(f"        main={ok_main} control={ok_ctrl} -> {sub_verdict}")
    print(f"  G1 overall: {verdict}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "results", "eac", "phaseA")
    runs = load_runs(root)
    if not runs:
        print("no runs found under", root)
        return
    by = rmse_table(runs)
    for sub in SUBSETS:
        lifetime_buckets(sub, runs)
    gate(by, runs)


if __name__ == "__main__":
    main()
