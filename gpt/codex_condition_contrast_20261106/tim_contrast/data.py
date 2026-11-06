"""Train-only, engine-disjoint data and strictly prefix-only history features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

from .vendor.cmapss import (load_cmapss_train, SELECTED_SENSOR_COLUMNS,
                            OperatingConditionScaler, SensorMinMaxScaler)

OUTER_SEED = 20261104
COLS = list(SELECTED_SENSOR_COLUMNS)
OPS = ["op_1", "op_2", "op_3"]
HGB_KW = dict(loss="squared_error", max_iter=200, learning_rate=0.05,
              max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=10.0,
              early_stopping=False)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_units(frame, fold):
    units = np.array(sorted(frame.unit.unique()), dtype=np.int64)
    if not 0 <= fold < 5:
        raise ValueError("fold must be 0..4")
    a, b = list(KFold(5, shuffle=True, random_state=OUTER_SEED).split(units))[fold]
    return units[a], units[b]


def measured_rows(frame, scaler):
    """Same 10 OPS + 84 sensor prefix summaries as the current FULL prior."""
    rows = []
    for unit, g in frame.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        x = scaler.transform(g).astype(np.float64)
        ops = g[OPS].to_numpy(np.float64)
        for c in range(50, min(len(g), 200) + 1, 10):
            recent, late = x[c-40:c], x[c-15:c]
            t = np.arange(40) - 19.5
            base = np.r_[c, ops[c-1], ops[:c].mean(0), ops[c-20:c].std(0)]
            sensor = np.r_[late.mean(0), recent.std(0), late.mean(0)-x[:15].mean(0),
                           t @ recent / (t @ t), x[:c].mean(0), x[:c].std(0)]
            rows.append((int(unit), c, int(g.cycle.max()), np.r_[base, sensor]))
    return rows


def fit_prior(frame, subset, train, valid, seed):
    tr = frame[frame.unit.isin(train)]
    va = frame[frame.unit.isin(valid)]
    return fit_prior_frames(tr, va, subset, seed)


def fit_prior_frames(tr, va, subset, seed):
    scaler = OperatingConditionScaler(COLS, n_clusters=6 if subset in ("FD002", "FD004") else 1,
                                      random_state=seed).fit(tr)
    a, b = measured_rows(tr, scaler), measured_rows(va, scaler)
    if not a:
        raise ValueError("Prior needs engines with at least 50 cycles")
    if not b:
        return {}
    counts = pd.Series([r[0] for r in a]).value_counts()
    model = HistGradientBoostingRegressor(random_state=seed, **HGB_KW)
    model.fit(np.stack([r[3] for r in a]), np.array([r[2] for r in a]),
              sample_weight=np.array([1 / counts[r[0]] for r in a]))
    pred = model.predict(np.stack([r[3] for r in b]))
    return {(r[0], r[1]): max(float(p), float(r[1])) for r, p in zip(b, pred)}


def nested_prior(frame, subset, train, valid, fold):
    table = {}
    train = np.array(sorted(train), dtype=np.int64)
    kf = KFold(4, shuffle=True, random_state=OUTER_SEED + 100 + fold)
    for k, (a, b) in enumerate(kf.split(train)):
        table.update(fit_prior(frame, subset, train[a], train[b], OUTER_SEED + 10*fold + k))
    table.update(fit_prior(frame, subset, train, valid, OUTER_SEED + fold))
    return table


def make_prefixes(frame, normalized, labels, pop_ref=None, endpoints_only=False, include_targets=True):
    """One query for EVERY observed t >= 50, with padded 100-cycle history.

Padding repeats the first frame for PE100, matching the upstream padding
convention. Padding is explicitly unobserved in the trend estimator. Labels
are built separately and never used in baseline/reference/history features.
"""
    fields = {k: [] for k in ("history", "cycles", "conditions", "observed", "as_of",
                              "extras", "unit", "target")}
    for unit, g in frame.groupby("unit", sort=True):
        idx = g.index.to_numpy()
        x = normalized[idx].astype(np.float32)
        c = g.cycle.to_numpy(np.int64)
        lab = labels[idx].astype(np.int64)
        if not np.array_equal(c, np.arange(1, len(g)+1)):
            raise ValueError(f"Unit {unit}: cycles must be contiguous and one-based")
        state = x[0].copy()
        for end in range(1, len(g)+1):
            state = 0.5*state + 0.5*x[end-1]
            if endpoints_only and end != len(g):
                continue
            if not endpoints_only and end < 50:
                continue
            if pop_ref is None:
                early = x[:min(5, end)]
                ref = early.mean(0)
                n = len(early)
                age = (n+1)/2.0
                std_ref = float(early.std(0).mean())
            else:
                m = int(lab[end-1])
                selected = np.flatnonzero((c[:end] <= 30) & (lab[:end] == m))[-5:]
                n = len(selected)
                mean = x[selected].mean(0) if n else np.zeros(x.shape[1], np.float32)
                age = float(c[selected].mean()) if n else 30.0
                std_ref = float(x[selected].std(0).mean()) if n > 1 else 0.0
                ref = n/(n+2.0)*mean + 2.0/(n+2.0)*pop_ref[m]
            deviation = float(np.abs(x[max(0, end-50):end] - ref).mean())
            jitter = float(np.abs(x[end-1] - state).mean())
            evidence = [min(1., n/5.), float(n >= 1), np.exp(-age/30.),
                        np.exp(-deviation), 1/(1+std_ref), jitter]
            start, count = max(0, end-100), min(end, 100)
            history = np.repeat(x[start:start+1], 100, axis=0)
            history[-count:] = x[start:end]
            times = np.zeros(100, np.int64)
            times[-count:] = c[start:end]
            conditions = np.full(100, -1, np.int64)
            conditions[-count:] = lab[start:end]
            observed = np.zeros_like(history, dtype=bool)
            observed[-count:] = True
            fields["history"].append(history)
            fields["cycles"].append(times)
            fields["conditions"].append(conditions)
            fields["observed"].append(observed)
            fields["as_of"].append(end)
            fields["extras"].append(np.r_[ref, min(end/250., 1.), evidence].astype(np.float32))
            fields["unit"].append(int(unit))
            fields["target"].append(min(125., float(len(g)-end)) if include_targets else np.nan)
    if not fields["target"]:
        raise ValueError("No engine has 50 observed cycles")
    out = {k: np.asarray(v) for k, v in fields.items()}
    out["target"] = out["target"].astype(np.float32)
    return out


def prepare_fold(data_root, subset, fold, use_prior=True):
    frame = load_cmapss_train(data_root, subset).sort_values(["unit", "cycle"]).reset_index(drop=True)
    train, valid = split_units(frame, fold)
    tr = frame[frame.unit.isin(train)].reset_index(drop=True)
    va = frame[frame.unit.isin(valid)].reset_index(drop=True)
    a, b = preprocess_frames(tr, va, subset, fold)
    table = nested_prior(frame, subset, train, valid, fold) if use_prior else None
    for data in (a, b):
        data["prior"] = np.zeros((len(data["target"]), 2), np.float32)
        if table is not None:
            data["prior"][:, 0] = [table[(int(u), min(int(t)//10*10, 200))]/250.
                                    for u, t in zip(data["unit"], data["as_of"])]
    meta = {"subset": subset, "fold": fold, "outer_seed": OUTER_SEED,
            "train_units": train.tolist(), "valid_units": valid.tolist(),
            "train_sha256": sha256(Path(data_root)/f"train_{subset}.txt"),
            "prior": "nested_FULL" if use_prior else "none",
            "prior_validation_lifetimes_used_in_fit": False,
            "n_train_queries": len(a["target"]), "n_valid_queries": len(b["target"])}
    return a, b, meta


def preprocess_frames(tr, va, subset, fold, endpoints_only=False):
    """Fit every transform on tr, sharing exact training/inference preprocessing."""
    tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
    multi = subset in ("FD002", "FD004")
    if multi:
        scaler = OperatingConditionScaler(COLS, n_clusters=6, random_state=7+fold).fit(tr)
        xt, xv = scaler.transform(tr), scaler.transform(va)
        # This second clustering deliberately matches run_ida.prep_fold.
        km = KMeans(n_clusters=6, random_state=7, n_init=10).fit(tr[OPS].to_numpy(np.float64))
        lt, lv = km.labels_, km.predict(va[OPS].to_numpy(np.float64))
        pop = np.zeros((6, len(COLS)), np.float32)
        for k in range(6):
            take = (tr.cycle.to_numpy() <= 15) & (lt == k)
            pop[k] = xt[take].mean(0) if take.any() else xt.mean(0)
    else:
        scaler = SensorMinMaxScaler(COLS).fit(tr[COLS].to_numpy(np.float64))
        xt, xv = scaler.transform(tr[COLS].to_numpy(np.float64)), scaler.transform(va[COLS].to_numpy(np.float64))
        lt, lv, pop = np.zeros(len(tr), np.int64), np.zeros(len(va), np.int64), None
    a = None if endpoints_only else make_prefixes(tr, xt, lt, pop)
    b = make_prefixes(va, xv, lv, pop, endpoints_only=endpoints_only, include_targets=not endpoints_only)
    return a, b


def save_cache(path, a, b, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".partial").open("wb") as stream:
        np.savez(stream, **{f"train_{k}": v for k, v in a.items()},
                 **{f"valid_{k}": v for k, v in b.items()}, metadata=json.dumps(meta))
    path.with_suffix(".partial").replace(path)


def load_cache(path):
    with np.load(path, allow_pickle=False) as z:
        a = {k[6:]: z[k] for k in z.files if k.startswith("train_")}
        b = {k[6:]: z[k] for k in z.files if k.startswith("valid_")}
        return a, b, json.loads(str(z["metadata"]))
