"""H1 arm: HGB directly predicts RUL with the same nested causal features as
the lifetime prior (Gate 1 comparison arm, runnable on CPU while GPUs serve
the Gate 0 queue).

Protocol parity with run_ida.py (which MUST NOT be modified while the queue
runs -- this is a separate file):
- identical outer 5-fold engine split (KFold shuffle seed 20261104 over
  sorted units);
- identical scalers (OperatingConditionScaler n_clusters=6 random_state=7+fold
  for FD002/FD004, SensorMinMaxScaler for FD001/FD003), fit on outer-train;
- identical window set: every cycle t in 50..T of each engine (PE window ends);
- identical target: piecewise-linear RUL capped at 125.

Causality: features are prefix_features(g, transformed, t) -- rows 1..t only.
HGB = 5-seed ensemble (seeds 11..55, same as the prior estimator), engine-
balanced sample weights 1/n_windows(unit), predictions averaged over seeds.
The only post-processing is max(pred, 0) (RUL is non-negative by definition,
the same information the lifetime estimator uses with max(pred, age+1)).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    SELECTED_SENSOR_COLUMNS, OperatingConditionScaler, SensorMinMaxScaler,
    build_piecewise_linear_rul, compute_rmse, load_cmapss_train,
)

OUTER_SEED = 20261104          # must match src/lifetime_prior/nested_prior.py
COLS = list(SELECTED_SENSOR_COLUMNS)
OPS = ("op_1", "op_2", "op_3")
PRIOR_SEEDS = (11, 22, 33, 44, 55)


def prefix_features_open(group, transformed: np.ndarray, t: int):
    """lsc_feasibility.prefix_features with c == len(group) allowed (the last
    cycle of a run-to-failure engine is a legitimate causal observation)."""
    x = transformed[:t].astype(np.float64)
    op = group.loc[:t - 1, list(OPS)].to_numpy(dtype=np.float64)
    recent = x[-min(t, 40):]
    early = x[: min(t, 15)]
    late = x[-min(t, 15):]
    time = np.arange(len(recent), dtype=np.float64)
    centred = time - time.mean()
    denom = max(float(np.square(centred).sum()), 1.0)
    slope = (centred[:, None] * (recent - recent.mean(0))).sum(0) / denom
    op_feat = np.concatenate([op[-1], op.mean(0), op[-min(t, 20):].std(0)])
    sensor_feat = np.concatenate([
        late.mean(0), recent.std(0), late.mean(0) - early.mean(0),
        slope, x.mean(0), x.std(0),
    ])
    return np.concatenate([[float(t)], op_feat, sensor_feat])


def outer_folds(subset, n_outer=5):
    df = load_cmapss_train(ROOT / "data" / "CMAPSS", subset)
    units = np.array(sorted(df["unit"].unique().tolist()), dtype=np.int64)
    kf = KFold(n_splits=n_outer, shuffle=True, random_state=OUTER_SEED)
    folds = []
    for ti, vi in kf.split(units):
        folds.append((set(int(u) for u in units[ti]),
                      set(int(u) for u in units[vi])))
    return folds, df


def window_rows(frame, transformed):
    """One row per window end t in 50..T: features, RUL target, (unit, t)."""
    feats, ys, meta = [], [], []
    for unit, g in frame.groupby("unit", sort=True):
        g = g.sort_values("cycle").reset_index(drop=True)
        cycles = g["cycle"].to_numpy(dtype=np.int64)
        if not np.array_equal(cycles, np.arange(1, len(g) + 1)):
            raise ValueError(f"non-contiguous cycles in unit {unit}")
        pos = g.index.to_numpy(dtype=np.int64)
        seq = transformed[pos]
        rul = build_piecewise_linear_rul(g, max_rul=125.0)
        for t in range(50, len(g) + 1):
            feats.append(prefix_features_open(g, seq, t))
            ys.append(float(rul[t - 1]))
            meta.append((int(unit), int(t)))
    return (np.stack(feats).astype(np.float64), np.asarray(ys, np.float64),
            np.asarray(meta, dtype=np.int64).reshape(-1, 2))


def fit_scalers(frame_tr, frame_va, subset, fold):
    """Same scaler rule as run_ida.prep_fold (fit on outer-train only)."""
    tr_df = frame_tr.sort_values(["unit", "cycle"]).reset_index(drop=True)
    va_df = frame_va.sort_values(["unit", "cycle"]).reset_index(drop=True)
    if subset in ("FD002", "FD004"):
        scaler = OperatingConditionScaler(COLS, n_clusters=6, random_state=7 + fold)
        tr_s = scaler.fit_transform(tr_df)
        va_s = scaler.transform(va_df)
    else:
        scaler = SensorMinMaxScaler(COLS)
        tr_s = scaler.fit_transform(tr_df[COLS].to_numpy(np.float64))
        va_s = scaler.transform(va_df[COLS].to_numpy(np.float64))
    return tr_df, va_df, np.asarray(tr_s, np.float64), np.asarray(va_s, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", default="experiments/ida/gate0")
    args = ap.parse_args()

    t0 = time.time()
    folds, df = outer_folds(args.subset)
    tr_u, va_u = folds[args.fold]
    tr_df_all = df[df.unit.isin(tr_u)]
    va_df_all = df[df.unit.isin(va_u)]

    tr_df, va_df, tr_s, va_s = fit_scalers(tr_df_all, va_df_all, args.subset, args.fold)
    Xtr, Ytr, Mtr = window_rows(tr_df, tr_s)
    Xva, Yva, Mva = window_rows(va_df, va_s)

    count = {}
    for u in Mtr[:, 0]:
        count[int(u)] = count.get(int(u), 0) + 1
    weights = np.array([1.0 / count[int(u)] for u in Mtr[:, 0]])

    preds = np.zeros(len(Xva), dtype=np.float64)
    for s in PRIOR_SEEDS:
        model = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=200, learning_rate=0.05,
            max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=10.0,
            early_stopping=False, random_state=s,
        )
        model.fit(Xtr, Ytr, sample_weight=weights)
        preds += model.predict(Xva)
        print(f"[H1 {args.subset} f{args.fold}] seed {s} done "
              f"({time.time() - t0:.0f}s)", flush=True)
    preds /= len(PRIOR_SEEDS)
    preds = np.maximum(preds, 0.0)          # RUL >= 0 by definition
    rmse = float(compute_rmse(Yva, preds))

    out_dir = ROOT / args.out / args.subset / f"H1_f{args.fold}_s0"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "unit": Mva[:, 0], "end_cycle": Mva[:, 1],
        "y_true": Yva, "y_pred": preds,
    }).to_csv(out_dir / "val_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "subset": args.subset, "arm": "H1", "fold": args.fold, "seed": 0,
        "best_val_rmse": rmse, "params": 0,
        "n_train_windows": len(Xtr), "n_val_windows": len(Xva),
        "n_val_units": len(set(Mva[:, 0].tolist())),
        "n_features": int(Xtr.shape[1]), "n_ensemble": len(PRIOR_SEEDS),
        "outer_seed": OUTER_SEED,
        "elapsed_seconds": time.time() - t0,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }, indent=2))
    print(f"BEST {args.subset} H1 f{args.fold} s0 val={rmse:.4f} "
          f"(windows {len(Xva)}, units {len(set(Mva[:,0].tolist()))})", flush=True)


if __name__ == "__main__":
    main()
