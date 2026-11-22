"""Paper C clean data pipeline: C-MAPSS, engine-level 5-fold CV, causal windows.

No test files are read here. Scalers/KMeans are fit on training engines only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
SENSOR_COLS = [f's{s}' for s in SENSORS]
CUTS = [0.25, 0.40, 0.55, 0.70, 0.85, 0.95]
CAP = 125.0


def load_train(subset: str, data_dir: str = '/home/user/rulb/data/CMAPSS') -> pd.DataFrame:
    path = f'{data_dir}/train_{subset}.txt'
    df = pd.read_csv(path, sep=r'\s+', header=None,
                     names=['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)])
    df = df.sort_values(['unit', 'cycle']).reset_index(drop=True)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    maxc = df.groupby('unit')['cycle'].transform('max')
    df['rul'] = np.minimum(maxc - df['cycle'], CAP)
    return df


def split_units(units, fold: int, n_splits: int = 5, seed: int = 2026):
    units = np.asarray(sorted(units))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(kf.split(units)):
        if i == fold:
            return set(units[tr]), set(units[va])
    raise ValueError(fold)


class Scaler:
    """Per-condition z-score; KMeans is fit on training engines only."""

    def __init__(self, subset: str, seed: int = 2026):
        self.subset = subset
        self.multi = subset in ('FD002', 'FD004')
        self.seed = seed
        self.kmeans = None
        self.means = {}
        self.stds = {}
        self.global_mean = None
        self.global_std = None

    def fit(self, df: pd.DataFrame):
        X = df[SENSOR_COLS].to_numpy(float)
        self.global_mean = X.mean(0)
        self.global_std = X.std(0) + 1e-6
        if self.multi:
            ops = df[['op1', 'op2', 'op3']].to_numpy(float)
            self.kmeans = KMeans(n_clusters=6, random_state=self.seed, n_init=10).fit(ops)
            df = df.copy()
            df['cluster'] = self.kmeans.predict(ops)
            for c in range(6):
                sub = df[df.cluster == c]
                if len(sub) < 10:
                    self.means[c] = self.global_mean
                    self.stds[c] = self.global_std
                else:
                    Xc = sub[SENSOR_COLS].to_numpy(float)
                    self.means[c] = Xc.mean(0)
                    self.stds[c] = Xc.std(0) + 1e-6
        else:
            self.means[0] = self.global_mean
            self.stds[0] = self.global_std
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.multi:
            df['cluster'] = self.kmeans.predict(df[['op1', 'op2', 'op3']].to_numpy(float))
        else:
            df['cluster'] = 0
        X = df[SENSOR_COLS].to_numpy(float)
        out = np.empty_like(X)
        for c in range(6):
            m = (df['cluster'] == c).to_numpy()
            if m.any():
                out[m] = (X[m] - self.means[c]) / self.stds[c]
        df[SENSOR_COLS] = out
        return df


def _window(arr, end, L):
    start = max(0, end - L + 1)
    w = arr[start:end + 1]
    pad = L - len(w)
    x = np.zeros((L, arr.shape[1]), dtype=np.float32)
    mask = np.zeros(L, dtype=np.float32)
    if pad:
        x[pad:] = w
        mask[pad:] = 1.0
    else:
        x[:] = w
        mask[:] = 1.0
    return x, mask


def make_train_windows(df, units, scaler, L, return_op=False):
    df = scaler.transform(df)
    xs, ms, ys, ops = [], [], [], []
    for u in sorted(units):
        g = df[df.unit == u]
        arr = g[SENSOR_COLS].to_numpy(float)
        op = g[['op1', 'op2', 'op3']].to_numpy(float)
        n = len(g)
        for t in range(n):
            x, m = _window(arr, t, L)
            xs.append(x); ms.append(m); ys.append(min(n - (t + 1), CAP))
            if return_op:
                start = max(0, t - L + 1)
                ops.append(op[start:t + 1].mean(0))
    out = (np.stack(xs).astype(np.float32), np.stack(ms).astype(np.float32),
           np.asarray(ys, np.float32))
    return out + (np.stack(ops).astype(np.float32),) if return_op else out


def _engine_base(arr, cl, K0=15):
    eb = np.zeros((6, arr.shape[1]), dtype=np.float32)
    for c in range(6):
        idx = np.where(cl[:K0] == c)[0]
        if len(idx):
            eb[c] = arr[idx].mean(0)
    return eb


def make_train_windows_meta(df, units, scaler, L, K0=15):
    df = scaler.transform(df)
    xs, ms, ys, cls, ebs = [], [], [], [], []
    for u in sorted(units):
        g = df[df.unit == u]
        arr = g[SENSOR_COLS].to_numpy(float)
        cl = g['cluster'].to_numpy(int)
        eb = _engine_base(arr, cl, K0)
        n = len(g)
        for t in range(n):
            x, m = _window(arr, t, L)
            start = max(0, t - L + 1)
            cw = cl[start:t + 1]
            cwin = np.full(L, -1, dtype=np.int64)
            cwin[L - len(cw):] = cw
            xs.append(x); ms.append(m); ys.append(min(n - (t + 1), CAP))
            cls.append(cwin); ebs.append(eb)
    return (np.stack(xs).astype(np.float32), np.stack(ms).astype(np.float32),
            np.asarray(ys, np.float32), np.stack(cls), np.stack(ebs))


def make_val_windows_meta(df, units, scaler, L, cuts=CUTS, K0=15):
    df = scaler.transform(df)
    xs, ms, ys, eng, cutv, cls, ebs = [], [], [], [], [], [], []
    for u in sorted(units):
        g = df[df.unit == u]
        arr = g[SENSOR_COLS].to_numpy(float)
        cl = g['cluster'].to_numpy(int)
        eb = _engine_base(arr, cl, K0)
        n = len(g)
        for frac in cuts:
            t = int(round(frac * (n - 1)))
            t = max(0, min(n - 1, t))
            x, m = _window(arr, t, L)
            start = max(0, t - L + 1)
            cw = cl[start:t + 1]
            cwin = np.full(L, -1, dtype=np.int64)
            cwin[L - len(cw):] = cw
            xs.append(x); ms.append(m)
            ys.append(min(n - (t + 1), CAP))
            eng.append(u); cutv.append(frac)
            cls.append(cwin); ebs.append(eb)
    return (np.stack(xs).astype(np.float32), np.stack(ms).astype(np.float32),
            np.asarray(ys, np.float32), np.asarray(eng), np.asarray(cutv),
            np.stack(cls), np.stack(ebs))


def make_val_windows(df, units, scaler, L, cuts=CUTS, return_op=False):
    df = scaler.transform(df)
    xs, ms, ys, eng, cutv, ops = [], [], [], [], [], []
    for u in sorted(units):
        g = df[df.unit == u]
        arr = g[SENSOR_COLS].to_numpy(float)
        op = g[['op1', 'op2', 'op3']].to_numpy(float)
        n = len(g)
        for frac in cuts:
            t = int(round(frac * (n - 1)))
            t = max(0, min(n - 1, t))
            x, m = _window(arr, t, L)
            xs.append(x); ms.append(m)
            ys.append(min(n - (t + 1), CAP))
            eng.append(u); cutv.append(frac)
            if return_op:
                start = max(0, t - L + 1)
                ops.append(op[start:t + 1].mean(0))
    out = (np.stack(xs).astype(np.float32), np.stack(ms).astype(np.float32),
           np.asarray(ys, np.float32), np.asarray(eng), np.asarray(cutv))
    return out + (np.stack(ops).astype(np.float32),) if return_op else out
