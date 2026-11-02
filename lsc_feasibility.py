#!/usr/bin/env python3
"""Leakage-safe, engine-disjoint total-lifetime identifiability gate for C-MAPSS.

Run from the deepseekrul repository root (PYTHONPATH=src):
  PYTHONPATH=src python lsc_feasibility.py --subsets FD002 --data data/CMAPSS
  PYTHONPATH=src python lsc_feasibility.py --subsets FD001 FD002 FD003 FD004

Uses *only* train_FDxxx.txt; never reads official test or RUL files.
Fixed observable-cycle cutoffs, not fractions of the engine's true lifetime.
Each validation unit belongs to exactly one development fold.
The result is a gate for the feasibility of estimating total lifetime, not a
claim that monotone HI or phase-conditioned architectures are novel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from dataset import OperatingConditionScaler, SELECTED_SENSOR_COLUMNS, load_cmapss_train

OPS = ("op_1", "op_2", "op_3")
COLS = tuple(SELECTED_SENSOR_COLUMNS)
ARMS = ("age", "age_ops", "sensor")


def prefix_features(group, transformed: np.ndarray, c: int):
    """Features from *exactly* rows 1..c, with no full-life normalisation."""
    assert c < len(group) and c >= 40
    x = transformed[:c].astype(np.float64)
    op = group.loc[:c - 1, list(OPS)].to_numpy(dtype=np.float64)
    recent = x[-min(c, 40):]
    early = x[: min(c, 15)]
    late = x[-min(c, 15):]
    time = np.arange(len(recent), dtype=np.float64)
    centred_time = time - time.mean()
    denom = max(float(np.square(centred_time).sum()), 1.0)
    slope = (centred_time[:, None] * (recent - recent.mean(0))).sum(0) / denom
    # Each component uses only measurements already collected by cycle c.
    op_feat = np.concatenate([op[-1], op.mean(0), op[-min(c, 20):].std(0)])
    sensor_feat = np.concatenate([
        late.mean(0), recent.std(0), late.mean(0) - early.mean(0),
        slope, x.mean(0), x.std(0),
    ])
    age = np.array([float(c)], dtype=np.float64)
    return age, np.concatenate([age, op_feat]), np.concatenate([age, op_feat, sensor_feat])


def make_rows(frame, scaler, cutoffs):
    row_list = []
    for unit, g in frame.groupby("unit", sort=True):
        g = g.sort_values("cycle").reset_index(drop=True)
        cycles = g["cycle"].to_numpy(dtype=np.int64)
        if not np.array_equal(cycles, np.arange(1, len(g) + 1)):
            raise ValueError(f"Non-contiguous 1-based cycles in unit={unit}")
        full_life = int(cycles[-1])
        normalized = scaler.transform(g)
        for c in cutoffs:
            if c >= full_life:
                continue  # No still-alive prefix at this time for this engine.
            xa, xo, xs = prefix_features(g, normalized, c)
            row_list.append(dict(unit=int(unit), c=int(c), true=full_life,
                                 age=xa, age_ops=xo, sensor=xs))
    return row_list


def fit_predict(train, valid, arm, seed):
    Xtr = np.stack([r[arm] for r in train])
    Ytr = np.array([r["true"] for r in train], dtype=float)
    Xva = np.stack([r[arm] for r in valid])
    count = {}
    for r in train:
        count[r["unit"]] = count.get(r["unit"], 0) + 1
    weights = np.array([1.0 / count[r["unit"]] for r in train])
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=200, learning_rate=0.05,
        max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=10.0,
        early_stopping=False, random_state=seed,
    )
    model.fit(Xtr, Ytr, sample_weight=weights)
    pred = model.predict(Xva)
    age = np.array([r["c"] for r in valid], dtype=float)
    return np.maximum(pred, age + 1.0)  # Existing survival is available information.


def safe_pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or min(a.std(), b.std()) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def weighted_metric(rows, indices, key, kind="mae"):
    """Engine-balanced mean errors for multiple time-cutoff observations."""
    sub = [rows[i] for i in indices]
    per_unit = {}
    for r in sub:
        err = float(r[key] - r["true"])
        per_unit.setdefault(r["unit"], []).append(abs(err) if kind == "mae" else err * err)
    vals = np.array([np.mean(v) for v in per_unit.values()])
    return float(vals.mean()) if kind == "mae" else float(np.sqrt(vals.mean()))


def paired_bootstrap(rows, a="age_ops", b="sensor", reps=2000, seed=2026):
    """Positive result means arm b has lower MAE. Resamples independent units."""
    units = np.unique([r["unit"] for r in rows])
    by_unit = {}
    for u in units:
        rr = [r for r in rows if r["unit"] == u]
        d = np.mean([abs(r[a] - r["true"]) - abs(r[b] - r["true"]) for r in rr])
        by_unit[int(u)] = float(d)
    v = np.array([by_unit[int(u)] for u in units])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(reps, len(v)))
    boot = v[idx].mean(axis=1)
    return dict(gain=float(v.mean()), ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
                n_engines=int(len(units)))


def summarize(rows, cutoffs, reps, seed):
    result = {"n_units": len(set(r["unit"] for r in rows)), "n_prefixes": len(rows), "by_cutoff": {}}
    for c in cutoffs:
        rr = [r for r in rows if r["c"] == c]
        if not rr:
            continue
        y = np.array([r["true"] for r in rr], float)
        m = {"n": len(rr), "age": c}
        for arm in ARMS:
            p = np.array([r[arm] for r in rr])
            m[arm] = dict(corr_T=safe_pearson(y, p), mae_T=float(np.mean(abs(p - y))),
                          rmse_T=float(np.sqrt(np.mean((p - y) ** 2))),
                          bias_T=float(np.mean(p - y)))
        m["sensor_gain_vs_age_ops"] = paired_bootstrap(rr, reps=reps, seed=seed)
        m["sensor_residual_corr"] = safe_pearson(
            y - np.array([r["age_ops"] for r in rr]),
            np.array([r["sensor"] - r["age_ops"] for r in rr]))
        result["by_cutoff"][str(c)] = m
    ids = np.arange(len(rows))
    result["overall_engine_balanced"] = {
        arm: dict(mae=weighted_metric(rows, ids, arm), rmse=weighted_metric(rows, ids, arm, "rmse"))
        for arm in ARMS
    }
    result["overall_engine_balanced"]["sensor_gain_vs_age_ops"] = paired_bootstrap(rows, reps=reps, seed=seed)
    return result


def run_one(subset, data_dir, cutoffs, n_splits, seed, reps):
    df = load_cmapss_train(data_dir, subset)
    units = np.array(sorted(df["unit"].unique().tolist()), dtype=np.int64)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    records = []
    for fold, (ti, vi) in enumerate(folds.split(units)):
        train_units = set(units[ti].tolist())
        val_units = set(units[vi].tolist())
        if train_units & val_units:
            raise AssertionError("Engine leak in split")
        train_df = df[df["unit"].isin(train_units)].copy()
        val_df = df[df["unit"].isin(val_units)].copy()
        scaler = OperatingConditionScaler(COLS, n_clusters=6 if subset in ("FD002", "FD004") else 1,
                                           random_state=seed + fold)
        scaler.fit(train_df)
        tr = make_rows(train_df, scaler, cutoffs)
        va = make_rows(val_df, scaler, cutoffs)
        if not tr or not va:
            raise ValueError("No eligible training/validation prefixes")
        pred = {arm: fit_predict(tr, va, arm, seed + fold) for arm in ARMS}
        for i, row in enumerate(va):
            records.append(dict(unit=row["unit"], fold=fold, c=row["c"], true=row["true"],
                                **{arm: float(pred[arm][i]) for arm in ARMS}))
    return records, summarize(records, cutoffs, reps, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CMAPSS", type=Path)
    ap.add_argument("--subsets", nargs="+", default=["FD002"])
    ap.add_argument("--cutoffs", nargs="+", type=int, default=[50, 75, 100, 125, 150, 175, 200])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("results/lsc_probe/feasibility.json"))
    args = ap.parse_args()
    cutoffs = sorted(set(args.cutoffs))
    if min(cutoffs) < 40:
        ap.error("Cutoffs must be >=40 to support the feature extractor")
    report = dict(protocol=dict(data="train-only", folds=args.folds, seed=args.seed,
                                cutoffs=cutoffs, target="true_total_lifetime_uncapped",
                                validation="engine-disjoint fixed observable cutoffs",
                                bootstrap_unit="engine", pipeline="HGB age / age+ops / age+ops+sensors",
                                note="Age-only and condition-only baselines are required; correlation alone is not a novelty gate"),
                  subsets={})
    records_all = {}
    for subset in args.subsets:
        subset = subset.upper()
        if subset not in ("FD001", "FD002", "FD003", "FD004"):
            ap.error(f"Invalid subset {subset}")
        records, summary = run_one(subset, args.data, cutoffs, args.folds, args.seed, args.bootstrap)
        report["subsets"][subset] = summary
        records_all[subset] = records
        print(f"{subset}: n_units={summary['n_units']} n_prefixes={summary['n_prefixes']}")
        for c, m in summary["by_cutoff"].items():
            print(f"  c={c:>3} n={m['n']:>3}  corr(T_hat,T)={m['sensor']['corr_T']} "
                  f"age+ops MAE={m['age_ops']['mae_T']:.2f} sensor MAE={m['sensor']['mae_T']:.2f} "
                  f"delta={m['sensor_gain_vs_age_ops']['gain']:+.2f}")
        print("  engine-balanced gain:", summary["overall_engine_balanced"]["sensor_gain_vs_age_ops"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    preds_file = args.out.with_name(args.out.stem + "_predictions.json")
    preds_file.write_text(json.dumps(records_all, ensure_ascii=False), encoding="utf-8")
    print("Wrote", args.out, "and", preds_file)


if __name__ == "__main__":
    main()
