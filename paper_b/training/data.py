"""C-MAPSS data pipeline for Paper B.

All preprocessing that touches labels or statistics is fitted on the training
engines of the current split only. The official test files are never opened by
the training path (only evaluation/evaluate_test.py loads them, and only after
the freeze manifest exists).

Protocol constants are in paper_b/configs/protocol.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_IDS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
SENSOR_COLS = [f's_{i}' for i in SENSOR_IDS]
OP_COLS = ['op_1', 'op_2', 'op_3']
COLUMNS = ['unit', 'cycle'] + OP_COLS + [f's_{i}' for i in range(1, 22)]
CYCLE_SCALE = 250.0
TAU = 2.0


def _read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r'\s+', header=None)
    if df.shape[1] != 26:
        raise ValueError(f'{path}: expected 26 columns, got {df.shape[1]}')
    df.columns = COLUMNS
    return df


def load_train(data_dir: Path, subset: str) -> pd.DataFrame:
    return _read_table(Path(data_dir) / f'train_{subset}.txt')


def load_test(data_dir: Path, subset: str):
    test = _read_table(Path(data_dir) / f'test_{subset}.txt')
    rul = np.loadtxt(Path(data_dir) / f'RUL_{subset}.txt', dtype=np.float64).reshape(-1)
    units = np.sort(test['unit'].unique())
    if len(units) != len(rul):
        raise ValueError(f'{subset}: {len(units)} test engines vs {len(rul)} RUL values')
    return test, dict(zip((int(u) for u in units), rul))


class OpScaler:
    """KMeans operating-condition clusters + cluster-wise sensor z-score."""

    def __init__(self, n_clusters=6, random_state=7):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.kmeans = None
        self.means = None
        self.scales = None

    def fit(self, frame: pd.DataFrame):
        from sklearn.cluster import KMeans
        ops = frame[OP_COLS].to_numpy(np.float64)
        sensors = frame[SENSOR_COLS].to_numpy(np.float64)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state,
                             n_init=20).fit(ops)
        labels = self.kmeans.labels_
        k = self.n_clusters
        self.means = np.zeros((k, sensors.shape[1]), np.float64)
        self.scales = np.ones((k, sensors.shape[1]), np.float64)
        gmean, gstd = sensors.mean(0), sensors.std(0)
        gscale = np.where(gstd > 0, gstd, 1.0)
        for c in range(k):
            m = labels == c
            if m.sum() < 2:
                self.means[c], self.scales[c] = gmean, gscale
            else:
                self.means[c] = sensors[m].mean(0)
                sd = sensors[m].std(0)
                self.scales[c] = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        labels = self.kmeans.predict(frame[OP_COLS].to_numpy(np.float64))
        sensors = frame[SENSOR_COLS].to_numpy(np.float64)
        out = (sensors - self.means[labels]) / self.scales[labels]
        return out.astype(np.float32)

    def labels(self, frame: pd.DataFrame) -> np.ndarray:
        return self.kmeans.predict(frame[OP_COLS].to_numpy(np.float64)).astype(np.int64)


def split_units(units, split_seed=7, fold=0, val_frac=0.2):
    units = np.sort(np.asarray(units))
    rng = np.random.default_rng(split_seed + 20261020)
    sh = rng.permutation(units)
    n_val = max(1, int(round(units.size * val_frac)))
    val = sorted(int(x) for x in sh[fold * n_val:(fold + 1) * n_val])
    train = sorted(int(x) for x in units if int(x) not in set(val))
    return train, val


def fit_hi(train_units, tr_frame, tr_scaled, seed=7):
    """PCA-1 health indicator fitted ONLY on this split's training engines.

    HI is a HEALTH index: it decreases with life (probe: per-engine Spearman
    vs cycle = -0.79..-0.92). The sign is forced negative, so the correct
    monotonic constraint is HI(t) <= HI(t-1).
    """
    from sklearn.cluster import KMeans
    d = tr_frame[tr_frame.unit.isin(train_units)].sort_values(['unit', 'cycle']).reset_index(drop=True)
    pos = d.index.to_numpy(np.int64)
    X = np.asarray(tr_scaled[pos], np.float64)
    ops = d[OP_COLS].to_numpy(np.float64)
    k = int(min(6, max(1, len(np.unique(ops, axis=0)))))
    lab = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(ops).labels_
    Z = X.copy()
    for c in range(k):
        m = lab == c
        if m.sum() >= 2:
            Z[m] = (Z[m] - Z[m].mean(0)) / (Z[m].std(0) + 1e-9)
    _, V = np.linalg.eigh(np.cov(Z.T))
    pc1 = Z @ V[:, -1]
    cyc = d['cycle'].to_numpy(float)
    if np.corrcoef(pc1, cyc)[0, 1] > 0:
        pc1 = -pc1
    lo, hi = np.percentile(pc1, 1), np.percentile(pc1, 99)
    val = np.clip((pc1 - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return {(int(u), int(c)): float(v)
            for u, c, v in zip(d['unit'].to_numpy(), cyc, val)}


def load_global_hi(data_dir: Path, subset: str):
    """Legacy global HI (leaky: fitted on all engines). Attribution only."""
    path = Path(data_dir) / 'HI' / f'hi_{subset}.csv'
    d = pd.read_csv(path)
    return {(int(u), int(c)): float(v)
            for u, c, v in zip(d['unit'], d['cycle'], d['hi_raw'])}


def _rul_curve(n, max_rul, legacy=False):
    r = (float(n) if legacy else float(n - 1)) - np.arange(n, dtype=np.float64)
    return np.clip(r, 0.0, max_rul)


def _baseline_ref(seq, clusters, cycles, t, pop_ref, tau=TAU):
    cl = int(clusters[t])
    early = np.where((cycles <= 30) & (clusters == cl) & (np.arange(len(seq)) <= t))[0]
    si = early[-5:] if early.size > 5 else early
    n = int(si.size)
    xbar = seq[si].mean(0) if n > 0 else np.zeros(seq.shape[1], np.float32)
    return ((n / (n + tau)) * xbar + (tau / (n + tau)) * pop_ref[cl]).astype(np.float32)


def _window(seq, end, span):
    m = min(end, span)
    x = np.zeros((span, seq.shape[1]), np.float32)
    x[span - m:] = seq[end - m:end]
    mask = np.zeros(span, np.float32)
    mask[span - m:] = 1.0
    return x, mask


def make_train_windows(frame, values, clusters, pop_ref, span, max_rul=125.0,
                       hi_map=None, legacy_rul=False):
    """Vectorised sliding-window construction (all cycle positions per engine)."""
    xs, ys, ms, bs, cs, cds, hs = [], [], [], [], [], [], []
    for uid, g in frame.groupby('unit', sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(values[pos], np.float32)
        cl = np.asarray(clusters[pos], np.int64)
        cyc = g['cycle'].to_numpy(np.int64)
        n = len(seq)
        padded = np.concatenate([np.zeros((span - 1, seq.shape[1]), np.float32), seq], axis=0)
        win = np.lib.stride_tricks.sliding_window_view(padded, span, axis=0)
        win = np.ascontiguousarray(win.transpose(0, 2, 1))          # (n, span, N)
        mask = np.zeros((n, span), np.float32)
        for t in range(n):
            m = min(t + 1, span)
            mask[t, span - m:] = 1.0
        refs = np.empty((n, seq.shape[1]), np.float32)
        for t in range(n):
            refs[t] = _baseline_ref(seq, cl, cyc, t, pop_ref)
        xs.append(win); ms.append(mask); ys.append(_rul_curve(n, max_rul, legacy_rul))
        bs.append(refs)
        cs.append(np.minimum(cyc / CYCLE_SCALE, 1.0).astype(np.float32))
        cds.append(cl)
        if hi_map:
            hs.append(np.asarray([hi_map.get((int(uid), int(c)), 0.0) for c in cyc], np.float32))
        else:
            hs.append(np.zeros(n, np.float32))
    return dict(x=np.concatenate(xs), y=np.concatenate(ys).astype(np.float32),
                mask=np.concatenate(ms), baseline=np.concatenate(bs),
                cycle=np.concatenate(cs).astype(np.float32),
                cond=np.concatenate(cds).astype(np.int64),
                hi=np.concatenate(hs).astype(np.float32))


def make_val_windows(frame, values, clusters, pop_ref, span, cuts=(0.4, 0.55, 0.7, 0.85, 0.95),
                     max_rul=125.0):
    """Engine-level truncated validation points.

    For each validation engine of length n, evaluate at cycle round(f*n) for
    f in cuts. True RUL at cycle c is n-c. The cut rule is a protocol constant;
    it never reads official test RUL values.
    """
    xs, ys, ms, bs, cs, cds, us, cts = [], [], [], [], [], [], [], []
    for uid, g in frame.groupby('unit', sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(values[pos], np.float32)
        cl = np.asarray(clusters[pos], np.int64)
        cyc = g['cycle'].to_numpy(np.int64)
        n = len(seq)
        seen = set()
        for f in cuts:
            c = int(min(n, max(1, round(f * n))))
            if c in seen:
                continue
            seen.add(c)
            t = c - 1
            x, mask = _window(seq, t + 1, span)
            xs.append(x); ms.append(mask)
            ys.append(float(min(max(n - c, 0), max_rul)))
            bs.append(_baseline_ref(seq, cl, cyc, t, pop_ref))
            cs.append(min(cyc[t] / CYCLE_SCALE, 1.0)); cds.append(int(cl[t]))
            us.append(int(uid)); cts.append(int(c))
    return dict(x=np.stack(xs), y=np.asarray(ys, np.float32), mask=np.stack(ms),
                baseline=np.stack(bs), cycle=np.asarray(cs, np.float32),
                cond=np.asarray(cds, np.int64), unit=np.asarray(us),
                cut=np.asarray(cts))


def make_test_windows(test_frame, values, clusters, pop_ref, span, rul_map, max_rul=125.0):
    xs, ms, bs, cs, cds, us, ys = [], [], [], [], [], [], []
    for uid, g in test_frame.groupby('unit', sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(values[pos], np.float32)
        cl = np.asarray(clusters[pos], np.int64)
        cyc = g['cycle'].to_numpy(np.int64)
        t = len(seq) - 1
        x, mask = _window(seq, t + 1, span)
        xs.append(x); ms.append(mask)
        bs.append(_baseline_ref(seq, cl, cyc, t, pop_ref))
        cs.append(min(cyc[t] / CYCLE_SCALE, 1.0)); cds.append(int(cl[t]))
        us.append(int(uid)); ys.append(float(min(rul_map[int(uid)], max_rul)))
    return dict(x=np.stack(xs), y=np.asarray(ys, np.float32), mask=np.stack(ms),
                baseline=np.stack(bs), cycle=np.asarray(cs, np.float32),
                cond=np.asarray(cds, np.int64), unit=np.asarray(us))


_CACHE = {}
_MAX_CACHE = 8


def _prepare_split(subset, data_dir, split_seed, fold, val_frac, n_cond):
    key = ('split', str(data_dir), subset, split_seed, fold, val_frac, n_cond)
    if key in _CACHE:
        return _CACHE[key]
    raw = load_train(data_dir, subset)
    units = np.sort(raw['unit'].unique())
    train_units, val_units = split_units(units, split_seed, fold, val_frac)
    tr = raw[raw.unit.isin(train_units)].sort_values(['unit', 'cycle']).reset_index(drop=True)
    va = raw[raw.unit.isin(val_units)].sort_values(['unit', 'cycle']).reset_index(drop=True)
    scaler = OpScaler(n_cond, split_seed).fit(tr)
    tr_s, va_s = scaler.transform(tr), scaler.transform(va)
    lt, lv = scaler.labels(tr), scaler.labels(va)
    pop = np.zeros((n_cond, len(SENSOR_COLS)), np.float32)
    early = tr['cycle'].to_numpy(np.int64) <= 15
    for k in range(n_cond):
        sel = early & (lt == k)
        pop[k] = tr_s[sel].mean(0) if sel.any() else tr_s.mean(0)
    out = dict(raw=raw, tr=tr, va=va, scaler=scaler, tr_s=tr_s, va_s=va_s,
               lt=lt, lv=lv, pop=pop, train_units=train_units, val_units=val_units)
    if len(_CACHE) >= _MAX_CACHE:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


def prepare_train(subset, data_dir, span=70, split_seed=7, fold=0, val_frac=0.2,
                  rul_cap=125.0, n_cond=6, cuts=(0.5, 0.75, 0.9), legacy_rul=False,
                  global_hi=False):
    """Build train + engine-level validation tensors. Never opens test files."""
    key = ('train', str(data_dir), subset, span, split_seed, fold, val_frac, rul_cap,
           n_cond, tuple(cuts), legacy_rul, global_hi)
    if key in _CACHE:
        return _CACHE[key]
    s = _prepare_split(subset, data_dir, split_seed, fold, val_frac, n_cond)
    hi_map = load_global_hi(data_dir, subset) if global_hi else fit_hi(
        s['train_units'], s['tr'], s['tr_s'], split_seed)
    trw = make_train_windows(s['tr'], s['tr_s'], s['lt'], s['pop'], span,
                             max_rul=rul_cap, hi_map=hi_map, legacy_rul=legacy_rul)
    vaw = make_val_windows(s['va'], s['va_s'], s['lv'], s['pop'], span,
                           cuts=cuts, max_rul=rul_cap)
    out = dict(train=trw, val=vaw, hi_map=hi_map, **{k: s[k] for k in
               ('raw', 'tr', 'va', 'scaler', 'tr_s', 'va_s', 'lt', 'lv', 'pop',
                'train_units', 'val_units')})
    if len(_CACHE) >= _MAX_CACHE:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


def prepare_test(subset, data_dir, span=70, split_seed=7, fold=0, val_frac=0.2,
                 rul_cap=125.0, n_cond=6):
    """Rebuild the training-fold preprocessing deterministically, then open test."""
    s = _prepare_split(subset, data_dir, split_seed, fold, val_frac, n_cond)
    test, rul_map = load_test(data_dir, subset)
    test = test.sort_values(['unit', 'cycle']).reset_index(drop=True)
    ts = s['scaler'].transform(test)
    lt_test = s['scaler'].labels(test)
    tw = make_test_windows(test, ts, lt_test, s['pop'], span, rul_map, max_rul=rul_cap)
    return dict(test=tw, rul_map=rul_map, **{k: s[k] for k in
                ('scaler', 'pop', 'train_units', 'val_units')})
