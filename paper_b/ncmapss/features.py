"""Flight-level feature construction for N-CMAPSS DS02.

Implements the frozen contract in ``docs/paper_b_ncmapss/DATASET_PROTOCOL.md``
§4.1 -- 51 columns per flight, identical for N1/N2/N3:

    X_s 14 sensors x {mean, std, OLS slope over the flight's 1 Hz samples} = 42
    W   4 columns   x {mean, std}                                          =  8
    age cycle / scale, scale = 95th percentile of the training fold's cycles = 1

``X_v``, ``T``, ``Y`` and ``A.hs`` are not accepted by this module at all: the
functions take ``X_s``, ``W`` and ``A`` only, and ``A`` is used solely for the
``(unit, cycle)`` grouping key. ``A.Fc`` and flight duration are not features.

The age scale is fitted per LOEO fold (on training engines only) via
:class:`AgeScaler`; nothing in this module looks at held-out or test units.

Sequence construction mirrors C-MAPSS ``paper_b/training/data.py::_window``:
one row per flight, ordered by ``cycle``, the model sees the last ``span``
flights ending at the prediction flight, right-aligned with left zero padding
and a validity mask.
"""
from __future__ import annotations

import numpy as np

SPAN = 70
N_SENSORS = 14
N_W = 4
N_STATS = 3
N_FEATURES = N_SENSORS * N_STATS + N_W * 2 + 1  # 51

SENSOR_FEATURE_NAMES = tuple(
    f'Xs_{s}_{stat}' for s in range(N_SENSORS) for stat in ('mean', 'std', 'slope'))
W_FEATURE_NAMES = tuple(f'W_{j}_{stat}' for j in range(N_W) for stat in ('mean', 'std'))
FEATURE_NAMES = SENSOR_FEATURE_NAMES + W_FEATURE_NAMES + ('age_scaled',)


class AgeScaler:
    """``cycle / q95(training cycles)``, fitted on the current fold's training engines."""

    def __init__(self, q=95.0):
        self.q = float(q)
        self.scale = 1.0

    def fit(self, cycles):
        c = np.asarray(cycles, dtype=np.float64)
        self.scale = float(np.percentile(c, self.q)) if c.size else 1.0
        self.scale = max(self.scale, 1.0)
        return self

    def transform(self, cycles):
        return (np.asarray(cycles, dtype=np.float64) / self.scale).astype(np.float32)


def _group_index(unit, cycle):
    """Stable group id per (unit, cycle), in first-appearance order, plus sorted order."""
    key = np.stack([np.asarray(unit, np.int64), np.asarray(cycle, np.int64)], axis=1)
    _, gid = np.unique(key, axis=0, return_inverse=True)
    gid = gid.reshape(-1)
    order = np.lexsort((cycle, unit))
    return gid, order


def _per_group_stats(gid, values, n_groups, min_samples=10):
    """mean, std and OLS slope of ``values`` against within-group sample index.

    ``values`` is (n_rows, k). The slope is fitted on the **centred** 1 Hz sample
    index (``t - mean(t)`` inside the flight), which is numerically stable and
    algebraically identical to fitting the raw index.

    A flight with fewer than ``min_samples`` samples gets ``slope = NaN`` rather
    than 0: 0 means "no change", NaN means "not estimable", and the difference
    matters because a short segment's slope is quadratically sensitive to its
    sample count. ``n_samples`` is returned so callers can report the drop.
    """
    n_rows, k = values.shape
    counts = np.bincount(gid, minlength=n_groups).astype(np.float64)
    order = np.argsort(gid, kind='stable')
    t_sorted = np.arange(n_rows) - np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]),
                                             counts.astype(np.int64))
    t = np.empty(n_rows, np.float64)
    t[order] = t_sorted
    # centre the sample index inside each flight before regressing
    t_c = t - (np.bincount(gid, weights=t, minlength=n_groups) / np.maximum(counts, 1))[gid]

    mean = np.empty((n_groups, k))
    std = np.empty((n_groups, k))
    slope = np.empty((n_groups, k))
    sum_tc2 = np.bincount(gid, weights=t_c * t_c, minlength=n_groups)
    estimable = counts >= min_samples
    for j in range(k):
        v = values[:, j].astype(np.float64)
        s = np.bincount(gid, weights=v, minlength=n_groups)
        s2 = np.bincount(gid, weights=v * v, minlength=n_groups)
        m = s / np.maximum(counts, 1)
        var = np.maximum(s2 / np.maximum(counts, 1) - m * m, 0.0)
        mean[:, j] = m
        std[:, j] = np.sqrt(var)
        s_tc_y = np.bincount(gid, weights=t_c * v, minlength=n_groups)
        with np.errstate(divide='ignore', invalid='ignore'):
            sl = s_tc_y / sum_tc2
        sl[~estimable] = np.nan          # too few samples: unknown, not zero
        slope[:, j] = sl
    return mean, std, slope, counts


def flight_table(W, X_s, A, min_samples=10, cond_labels=None, n_cond=6):
    """Build the per-flight statistics table from raw 1 Hz arrays.

    Parameters
    ----------
    W : (n, 4) scenario descriptors
    X_s : (n, 14) measured sensors
    A : (n, 4) auxiliary columns in documented order ``unit, cycle, Fc, hs``
    min_samples : flights with fewer 1 Hz samples get ``slope = NaN`` and
        ``low_confidence = True``

    Returns
    -------
    dict with ``unit``, ``cycle``, ``features`` (n_flights, 50), ``n_samples``,
    ``low_confidence``, ``feature_names``, ``rows_per_flight`` and ``row_index``.
    ``n_samples`` is a diagnostic column and is **not** part of the 51 features.
    """
    W = np.asarray(W, np.float64)
    X_s = np.asarray(X_s, np.float64)
    A = np.asarray(A, np.float64)
    if W.shape[1] != N_W or X_s.shape[1] != N_SENSORS:
        raise ValueError(f'expected W (n,{N_W}) and X_s (n,{N_SENSORS}), '
                         f'got {W.shape} and {X_s.shape}')
    unit = A[:, 0].astype(np.int64)
    cycle = A[:, 1].astype(np.int64)

    gid, order = _group_index(unit, cycle)
    # sort rows by (unit, cycle), then renumber groups so that ids are contiguous
    # in the sorted order (the cumulative-count boundaries below rely on that)
    gid_sorted = gid[order]
    _, gid_sorted = np.unique(gid_sorted, return_inverse=True)
    gid_sorted = gid_sorted.reshape(-1)
    n_groups = int(gid_sorted.max()) + 1
    W_s, X_sorted = W[order], X_s[order]

    m_x, s_x, sl_x, n_x = _per_group_stats(gid_sorted, X_sorted, n_groups, min_samples)
    m_w, s_w, _, _ = _per_group_stats(gid_sorted, W_s, n_groups, min_samples)

    feats = np.concatenate([m_x, s_x, sl_x, m_w, s_w], axis=1).astype(np.float32)
    # one (unit, cycle) per group, in sorted order
    first = np.concatenate([[0], np.cumsum(np.bincount(gid_sorted, minlength=n_groups))[:-1]])
    unit_f = unit[order][first]
    cycle_f = cycle[order][first]
    row_index = order[first]          # original row of each flight's first 1 Hz sample
    n_samples = n_x.astype(np.int64)
    out = dict(unit=unit_f, cycle=cycle_f, features=feats,
               feature_names=SENSOR_FEATURE_NAMES + W_FEATURE_NAMES,
               rows_per_flight=np.bincount(gid_sorted, minlength=n_groups),
               row_index=row_index, n_samples=n_samples,
               low_confidence=n_samples < min_samples)
    if cond_labels is not None:
        # per-flight operating-condition id = mode of its 1 Hz cluster labels.
        # This is a re-encoding of W, which is already in the shared 51 columns;
        # it adds no information the Vanilla rung does not receive.
        lab = np.asarray(cond_labels, np.int64)[order]
        code = gid_sorted.astype(np.int64) * n_cond + lab
        counts = np.bincount(code, minlength=n_groups * n_cond).reshape(n_groups, n_cond)
        out['cond'] = counts.argmax(1).astype(np.int64)
    return out


_PER_FLIGHT_KEYS = ('unit', 'cycle', 'features', 'row_index', 'n_samples',
                    'low_confidence', 'cond')


def filter_low_confidence(table):
    """Drop flights whose slope is not estimable, returning the kept table and a log.

    Dropping is explicit and counted; nothing is silently discarded. The keep
    mask is returned in ``info`` so per-flight labels can be filtered with it.
    """
    keep = ~np.asarray(table['low_confidence'], bool)
    out = {}
    for k, v in table.items():
        if (k in _PER_FLIGHT_KEYS and isinstance(v, np.ndarray)
                and v.shape[0] == len(keep)):
            out[k] = v[keep]
        else:
            out[k] = v
    dropped = np.flatnonzero(~keep)
    info = dict(n_total=int(len(keep)), n_kept=int(keep.sum()), n_dropped=int(dropped.size),
                keep=keep,
                dropped=[(int(table['unit'][i]), int(table['cycle'][i]),
                          int(table['n_samples'][i])) for i in dropped])
    return out, info


def build_matrix(table, age_scaler):
    """Append the scaled age column -> (n_flights, 51) float32."""
    age = age_scaler.transform(table['cycle']).reshape(-1, 1)
    X = np.concatenate([table['features'], age], axis=1).astype(np.float32)
    if X.shape[1] != N_FEATURES:
        raise AssertionError(f'feature matrix has {X.shape[1]} columns, contract says {N_FEATURES}')
    return X


def make_windows(X, unit, cycle, y=None, span=SPAN):
    """Sliding windows of the last ``span`` flights, right-aligned, left zero-padded.

    Returns ``x`` (n_flights, span, n_features), ``mask`` (n_flights, span) and,
    when ``y`` is given, ``y`` aligned to the window end. ``unit``/``cycle`` of
    the prediction flight are returned for evaluation.
    """
    X = np.asarray(X, np.float32)
    n, k = X.shape
    out_x = np.zeros((n, span, k), np.float32)
    out_m = np.zeros((n, span), np.float32)
    for u in np.unique(unit):
        idx = np.flatnonzero(unit == u)
        idx = idx[np.argsort(cycle[idx], kind='stable')]
        seq = X[idx]
        for j in range(len(idx)):
            end = j + 1
            start = max(0, end - span)
            m = end - start
            out_x[idx[j], span - m:] = seq[start:end]
            out_m[idx[j], span - m:] = 1.0
    res = dict(x=out_x, mask=out_m, unit=np.asarray(unit), cycle=np.asarray(cycle))
    if y is not None:
        y = np.asarray(y, np.float32)
        if len(y) != n:
            raise ValueError(f'label length {len(y)} != number of flights {n}; '
                             'pass one label per flight, not one per 1 Hz sample')
        res['y'] = y
    return res
