"""LSC Phase-2 data pipeline: engine-disjoint folds, windows, labels.

Protocol is frozen in paper/LSC_DESIGN.md section 4.  Only train_FDxxx.txt is
ever read; engines run to failure inside the train file, so uncapped lifetime
labels are legal for training and development evaluation.  Official test data
is never touched here.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from dataset import OperatingConditionScaler, load_cmapss_train

WINDOW = 40
HIST = 200          # long-history span for the scale head
HLEN = (HIST + 3) // 4   # strided length, fixed at 50
FEATURES = tuple(f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21))
OPS = ("op_1", "op_2", "op_3")
NF = len(FEATURES) + len(OPS)


def fleet_folds(subset: str, n_splits: int = 5, seed: int = 2026):
    df = load_cmapss_train("data/CMAPSS", subset)
    units = np.array(sorted(df["unit"].unique().tolist()), dtype=np.int64)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return df, units, list(folds.split(units))


def build_windows(df, unit_ids, scaler):
    """Windows of length WINDOW ending at every observed cycle.

    Returns dict with per-window arrays.  rul = min(125, T - end) is the
    standard capped RUL target; life = uncapped T is the scale-head target
    (legal: every engine in the train file runs to failure); cyc is the
    absolute end cycle, needed by the oracle arm and the truncation term.
    Xs is the long strided history (up to HIST cycles, stride 4) that the
    lifetime-scale head consumes: the stage-1 gate measured that lifetime
    information grows with prefix length, so a 40-row window alone starves it.
    """
    X, Xs, rul, life, cyc, eng = [], [], [], [], [], []
    for u in unit_ids:
        g = df[df["unit"] == u].sort_values("cycle")
        z = scaler.transform(g)
        x = np.concatenate([z, g.loc[:, list(OPS)].to_numpy(np.float32)], axis=1)
        T = int(g["cycle"].iloc[-1])
        n = len(g)
        if n < WINDOW:
            continue
        for end in range(WINDOW, n + 1):
            X.append(x[end - WINDOW:end])
            start = max(0, end - 1 - HIST)
            hist = x[start:end:4]
            if len(hist) < HLEN:               # front-pad by repeating the first row
                pad = np.repeat(hist[:1], HLEN - len(hist), axis=0)
                hist = np.concatenate([pad, hist], axis=0)
            Xs.append(hist[-HLEN:])
            rul.append(min(125.0, T - end))
            life.append(float(T))
            cyc.append(end)
            eng.append(int(u))
    return dict(X=np.stack(X).astype(np.float32),
                Xs=np.stack(Xs).astype(np.float32),
                rul=np.array(rul, np.float32),
                life=np.array(life, np.float32),
                cyc=np.array(cyc, np.int64),
                eng=np.array(eng, np.int64))


def make_fold_data(subset: str, fold: int, seed: int = 2026):
    df, units, splits = fleet_folds(subset, seed=seed)
    ti, vi = splits[fold]
    tr_units, va_units = units[ti], units[vi]
    n_clusters = 6 if subset in ("FD002", "FD004") else 1
    scaler = OperatingConditionScaler(FEATURES, n_clusters=n_clusters,
                                      random_state=seed + fold)
    train_df = df[df["unit"].isin(tr_units)]
    scaler.fit(train_df)
    tr = build_windows(train_df, tr_units, scaler)
    va = build_windows(df[df["unit"].isin(va_units)], va_units, scaler)
    # lifetime quartiles of the training fleet define the bias buckets
    life_by_unit = df.groupby("unit")["cycle"].max().loc[tr_units].to_numpy()
    edges = np.quantile(life_by_unit, [0.25, 0.5, 0.75])
    return tr, va, scaler, edges


def nasa_score(err):
    """Standard NASA C-MAPSS scoring function on signed errors."""
    err = np.asarray(err, np.float64)
    return float(np.sum(np.where(err < 0, np.exp(-err / 13.0) - 1.0,
                                 np.exp(err / 10.0) - 1.0)))
