"""N-CMAPSS clean data pipeline: per-cycle aggregates, unit-level 5-fold, causal windows."""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

FILES = {
    'DS02': 'N-CMAPSS_DS02-006.h5',
    'DS05': 'N-CMAPSS_DS05.h5',
    'DS06': 'N-CMAPSS_DS06.h5',
    'DS07': 'N-CMAPSS_DS07.h5',
}
ROOT = '/var/tmp/ncmapss_offload_20261120/extracted'
CUTS = [0.25, 0.40, 0.55, 0.70, 0.85, 0.95]
SENSOR_COLS = [f'x{i}' for i in range(14)]
W_COLS = [f'w{i}' for i in range(4)]
FEATS = SENSOR_COLS + W_COLS


def load_subset(subset: str) -> pd.DataFrame:
    path = f'{ROOT}/{FILES[subset]}'
    with h5py.File(path, 'r') as f:
        A = f['A_dev'][:]
        X = f['X_s_dev'][:]
        W = f['W_dev'][:]
        Y = f['Y_dev'][:, 0]
    df = pd.DataFrame(A[:, :2], columns=['unit', 'cycle'])
    for i in range(14):
        df[f'x{i}'] = X[:, i]
    for i in range(4):
        df[f'w{i}'] = W[:, i]
    df['y'] = Y
    df = df.groupby(['unit', 'cycle'], as_index=False).mean()
    return df.sort_values(['unit', 'cycle']).reset_index(drop=True)


def split_units(units, fold, n_splits=5, seed=2026):
    units = np.asarray(sorted(units))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(kf.split(units)):
        if i == fold:
            return set(units[tr]), set(units[va])
    raise ValueError(fold)


class Scaler:
    def __init__(self, seed=2026):
        self.mu = None
        self.sd = None
        self.kmeans = None
        self.seed = seed

    def fit(self, df):
        X = df[FEATS].to_numpy(float)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        W = df[W_COLS].to_numpy(float)
        self.kmeans = KMeans(n_clusters=6, random_state=self.seed, n_init=10).fit(W)
        return self

    def transform(self, df):
        df = df.copy()
        df[FEATS] = (df[FEATS].to_numpy(float) - self.mu) / self.sd
        df['cluster'] = self.kmeans.predict(df[W_COLS].to_numpy(float))
        return df


def _window(arr, end, L):
    start = max(0, end - L + 1)
    w = arr[start:end + 1]
    pad = L - len(w)
    x = np.zeros((L, arr.shape[1]), dtype=np.float32)
    m = np.zeros(L, dtype=np.float32)
    if pad:
        x[pad:] = w; m[pad:] = 1.0
    else:
        x[:] = w; m[:] = 1.0
    return x, m


def _engine_base(arr, cl, K0=15):
    eb = np.zeros((6, arr.shape[1]), dtype=np.float32)
    for c in range(6):
        idx = np.where(cl[:K0] == c)[0]
        if len(idx):
            eb[c] = arr[idx].mean(0)
    return eb


def make_windows(df, units, scaler, L, cuts=None):
    df = scaler.transform(df)
    xs, ms, ys, eng, cv = [], [], [], [], []
    for u in sorted(units):
        g = df[df.unit == u]
        arr = g[FEATS].to_numpy(float)
        y = g['y'].to_numpy(float)
        n = len(g)
        idxs = list(range(n)) if cuts is None else [
            max(0, min(n - 1, int(round(f * (n - 1))))) for f in cuts]
        for j, t in enumerate(idxs):
            x, m = _window(arr, t, L)
            xs.append(x); ms.append(m); ys.append(y[t])
            eng.append(u)
            cv.append(-1.0 if cuts is None else cuts[j])
    return (np.stack(xs).astype(np.float32), np.stack(ms).astype(np.float32),
            np.asarray(ys, np.float32), np.asarray(eng), np.asarray(cv))


def make_windows_meta(df, units, scaler, L, cuts=None, K0=15):
    df = scaler.transform(df)
    xs, ms, ys, eng, cv, cls, ebs, ws = [], [], [], [], [], [], [], []
    for u in sorted(units):
        g = df[df.unit == u]
        arr = g[FEATS].to_numpy(float)
        w = g[W_COLS].to_numpy(float)
        cl = g['cluster'].to_numpy(int)
        y = g['y'].to_numpy(float)
        eb = _engine_base(arr, cl, K0)
        n = len(g)
        idxs = list(range(n)) if cuts is None else [
            max(0, min(n - 1, int(round(f * (n - 1))))) for f in cuts]
        for j, t in enumerate(idxs):
            x, m = _window(arr, t, L)
            start = max(0, t - L + 1)
            cw = cl[start:t + 1]
            cwin = np.full(L, -1, dtype=np.int64)
            cwin[L - len(cw):] = cw
            ww = np.zeros((L, 4), dtype=np.float32)
            ww[L - len(cw):] = w[start:t + 1]
            xs.append(x); ms.append(m); ys.append(y[t])
            eng.append(u)
            cv.append(-1.0 if cuts is None else cuts[j])
            cls.append(cwin); ebs.append(eb); ws.append(ww)
    return (np.stack(xs).astype(np.float32), np.stack(ms).astype(np.float32),
            np.asarray(ys, np.float32), np.asarray(eng), np.asarray(cv),
            np.stack(cls), np.stack(ebs), np.stack(ws))
