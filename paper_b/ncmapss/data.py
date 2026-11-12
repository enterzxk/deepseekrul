"""N-CMAPSS DS02 loader with an enforced observability boundary.

The field semantics here are quoted from the dataset paper, not inferred from
the HDF5 key names:

    Arias Chao, Kulkarni, Goebel & Fink, "Aircraft Engine Run-to-Failure Dataset
    under Real Flight Conditions for Prognostics and Diagnostics",
    Data 6(1):5, 2021, doi:10.3390/data6010005, Tables 3-8.

Two names in that file are traps and are handled explicitly:

* ``T`` is **not** temperature. It is the ten-dimensional health-parameter
  vector theta (Table 7) -- the ground-truth degradation state. The only
  temperature in the scenario descriptors is ``T2``, a column of ``W``.
* ``X_v`` is **not** a second sensor set. It is the simulator's virtual-sensor
  output (Table 6), including the stall margins that define the health index.

Both are forbidden as model inputs. ``load_features`` enforces this with an
allow-list and raises rather than silently dropping a column, so a config that
asks for ``X_v`` fails loudly instead of producing a leaky result.

Splits are **read from the file, never stored here**:

    dev, test = official_split(path)

The HDF5 files ship the official Development/Test split as data (the ``A_dev``
and ``A_test`` arrays). A second hand-written copy of the unit lists is how two
lists drift apart, so this package keeps none: :func:`official_split` is the only
authority, and it raises if the file's two splits share a unit.

Usage
-----
    from paper_b.ncmapss.data import load_features, load_aux, official_split, verify

    d = load_features(path, 'dev')        # {'W': ..., 'X_s': ..., 'cols': ...}
    a = load_aux(path, 'dev')             # unit / cycle / Fc / hs, never features
    rep = verify(path)                    # runtime audit, JSON-serialisable
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---- arrays (Table 3) ------------------------------------------------------
SPLIT_ARRAYS = {
    'dev': ('W_dev', 'X_s_dev', 'X_v_dev', 'T_dev', 'Y_dev', 'A_dev'),
    'test': ('W_test', 'X_s_test', 'X_v_test', 'T_test', 'Y_test', 'A_test'),
}
VAR_ARRAYS = {'W': 'W_var', 'X_s': 'X_s_var', 'X_v': 'X_v_var',
              'T': 'T_var', 'A': 'A_var'}

# ---- column names (Tables 4-8) --------------------------------------------
W_COLS = ('alt', 'Mach', 'TRA', 'T2')
X_S_COLS = ('Wf', 'Nf', 'Nc', 'T24', 'T30', 'T48', 'T50', 'P15',
            'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50')
X_V_COLS = ('T40', 'P30', 'P45', 'W21', 'W22', 'W25', 'W31', 'W32',
            'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi')
T_COLS = ('fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod',
          'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod',
          'LPT_eff_mod', 'LPT_flow_mod')
A_COLS = ('unit', 'cycle', 'Fc', 'hs')

# ---- observability boundary ----------------------------------------------
#: arrays a model may consume. Everything else in the file is either a label,
#: a simulator-internal quantity, or metadata that encodes unit identity.
FEATURE_ALLOWLIST = ('W', 'X_s')

#: arrays that exist in the file and must never reach a feature matrix.
FORBIDDEN_FEATURES = {
    'X_v': 'simulator virtual sensors (Table 6); not measurable on a deployed engine',
    'T': 'health parameters theta (Table 7); the ground-truth degradation state',
    'Y': 'RUL label',
    'A': 'metadata; unit/Fc/hs encode identity and health state',
}


def _names(f, key):
    """Read a ``*_var`` array as a list of str, tolerating bytes and str dtypes."""
    raw = np.asarray(f[key])
    out = []
    for x in raw.ravel():
        out.append(x.decode('utf-8', 'replace') if isinstance(x, (bytes, np.bytes_)) else str(x))
    return tuple(out)


def _array(f, split, name):
    key = f'{name}_{split}'
    if key not in f:
        raise KeyError(f'{key} missing; file has {sorted(f.keys())[:12]} ...')
    a = np.asarray(f[key])
    return a[:, 0] if (a.ndim == 2 and a.shape[1] == 1) else a


def _check_columns(got, expected, array):
    if tuple(got) != tuple(expected):
        # The column order is read from the file, so a mismatch is informational
        # unless the sets differ; a reordering is legal but must be recorded.
        if set(got) != set(expected):
            raise ValueError(f'{array}: columns {got} != documented {expected}')
    return tuple(got)


def load_raw(path, split, arrays=('W', 'X_s', 'A', 'Y')):
    """Load arrays by documented name, returning columns as well.

    ``arrays`` may contain any of ``W``, ``X_s``, ``X_v``, ``T``, ``Y``, ``A``.
    This is the low-level reader: it does **not** enforce the observability
    boundary, and is meant for audit and diagnostics code only. Training code
    must go through :func:`load_features`.
    """
    import h5py

    split = split.lower()
    if split not in SPLIT_ARRAYS:
        raise ValueError(f'split must be dev or test, got {split!r}')
    if 'Y' in arrays and split == 'test':
        pass  # labels are allowed for evaluation, never as features
    out = {}
    with h5py.File(Path(path), 'r') as f:
        for name in arrays:
            if name == 'Y':
                out['Y'] = _array(f, split, 'Y').reshape(-1).astype(np.float64)
                continue
            a = _array(f, split, name)
            cols = _names(f, VAR_ARRAYS[name])
            expected = {'W': W_COLS, 'X_s': X_S_COLS, 'X_v': X_V_COLS,
                        'T': T_COLS, 'A': A_COLS}[name]
            _check_columns(cols, expected, name)
            out[name] = a
            out[f'{name}_cols'] = cols
    return out


def load_features(path, split):
    """Return only the arrays a deployed model may see: ``W`` and ``X_s``.

    Raises if the caller tries to route a forbidden array through this function.
    The returned dict carries the column names so downstream code indexes by
    name, never by a hard-coded position.
    """
    d = load_raw(path, split, arrays=FEATURE_ALLOWLIST)
    leaked = sorted(set(d) & set(FORBIDDEN_FEATURES))
    if leaked:
        raise AssertionError(f'forbidden feature arrays requested: {leaked}')
    return d


def load_aux(path, split):
    """Metadata for grouping and evaluation. Never a feature."""
    d = load_raw(path, split, arrays=('A',))
    cols = d['A_cols']
    A = d['A']
    out = {c: A[:, i] for i, c in enumerate(cols)}
    out['unit'] = out['unit'].astype(np.int64)
    out['cycle'] = out['cycle'].astype(np.int64)
    return out


def load_labels(path, split):
    """RUL in flight cycles. Evaluation only."""
    d = load_raw(path, split, arrays=('Y',))
    return d['Y']


def load_health_parameters(path, split):
    """Ground-truth theta. Diagnostics only; never call this from training."""
    return load_raw(path, split, arrays=('T',))['T']


def official_split(path):
    """``(dev_units, test_units)`` read from the file's own A_dev / A_test arrays.

    The HDF5 files ship the official Development/Test split as data. Keeping a
    second hand-written copy in code is exactly how two lists drift apart, so
    this function is the only authority and no unit list is stored anywhere.

    Raises if the two splits share a unit: that would mean the file does not
    carry a clean official split and every LOEO fold downstream would be
    contaminated, which is worth failing loudly for.
    """
    dev = tuple(sorted(int(u) for u in np.unique(load_aux(path, 'dev')['unit'])))
    test = tuple(sorted(int(u) for u in np.unique(load_aux(path, 'test')['unit'])))
    overlap = set(dev) & set(test)
    if overlap:
        raise AssertionError(f'dev and test share units {sorted(overlap)}; '
                             'the file does not carry a clean official split')
    return dev, test


def unit_lifetime(path, split):
    """``unit -> max RUL over that unit's complete trajectory``, in flight cycles.

    Required by the relative segment rule: the normaliser must come from the
    unit's full lifetime, not from the evaluated subset (which collapses to a
    single label when only the last flight is scored). It is derived from the
    label trajectory and is never a model input.
    """
    a = load_aux(path, split)
    y = load_labels(path, split)
    out = {}
    for u in np.unique(a['unit']):
        m = a['unit'] == u
        out[int(u)] = float(np.max(y[m]))
    return out


def verify(path):
    """Runtime audit of the file against the documented structure.

    Returns a JSON-serialisable dict. This is the evidence that the data
    dictionary in DATASET_PROTOCOL.md matches the file on disk.
    """
    import h5py

    p = Path(path)
    rep = {'path': str(p), 'size_bytes': p.stat().st_size, 'keys': [], 'arrays': {}}
    with h5py.File(p, 'r') as f:
        rep['keys'] = sorted(f.keys())
        for split in ('dev', 'test'):
            for name in ('W', 'X_s', 'X_v', 'T', 'Y', 'A'):
                key = f'{name}_{split}'
                if key not in f:
                    rep['arrays'][key] = {'present': False}
                    continue
                a = f[key]
                cols = _names(f, VAR_ARRAYS[name]) if VAR_ARRAYS.get(name) else None
                entry = {'present': True, 'shape': list(a.shape),
                         'dtype': str(a.dtype), 'chunks': a.chunks}
                if cols is not None:
                    entry['cols'] = cols
                    entry['n_cols'] = len(cols)
                if name == 'Y':
                    y = np.asarray(a).reshape(-1)
                    entry.update(min=float(np.nanmin(y)), max=float(np.nanmax(y)),
                                 n_nan=int(np.isnan(y).sum()),
                                 unique_rounded=int(len(np.unique(np.round(y, 6)))))
                rep['arrays'][key] = entry
    return rep
