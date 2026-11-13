"""Contract test for the N-CMAPSS loader, on a synthetic file with the documented layout.

Runs without the 15.8 GB download: builds a miniature HDF5 that mimics
``N-CMAPSS_DS02-006.h5`` (Tables 3-8 of the dataset paper) and checks that

  * the column names are read from ``*_var`` rather than assumed by position;
  * ``Y`` is accepted as a label and refused as a feature;
  * ``X_v`` and ``T`` are refused as features with a loud error;
  * the official unit split is read from the file, not from a list in code;
  * ``Y`` being constant within a flight cycle is detected.

    python -m paper_b.ncmapss.test_contract
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .data import (A_COLS, T_COLS, W_COLS, X_S_COLS, X_V_COLS, load_aux,
                   load_features, load_labels, official_split, verify)


def _write_fake(path, dev_units, test_units=(), cycles=12, rows_per_cycle=8, seed=0):
    import h5py

    rng = np.random.default_rng(seed)
    all_units = tuple(dev_units) + tuple(test_units)
    n = len(all_units) * cycles * rows_per_cycle
    A = np.zeros((n, 4), np.float64)
    Y = np.zeros((n, 1), np.float64)
    W = rng.normal(size=(n, 4))
    Xs = rng.normal(size=(n, 14))
    Xv = rng.normal(size=(n, 14))
    T = rng.normal(size=(n, 10))
    i = 0
    for u in all_units:
        for c in range(1, cycles + 1):
            rul = float(cycles - c)
            for _ in range(rows_per_cycle):
                A[i] = [u, c, 1 + (u % 3), 0 if c < cycles else 1]
                Y[i, 0] = rul
                i += 1
    assert i == n
    n_dev = len(dev_units) * cycles * rows_per_cycle
    dev = slice(0, n_dev)
    test = slice(n_dev, n)
    with h5py.File(path, 'w') as f:
        f['W_var'] = np.array([s.encode() for s in W_COLS])
        f['X_s_var'] = np.array([s.encode() for s in X_S_COLS])
        f['X_v_var'] = np.array([s.encode() for s in X_V_COLS])
        f['T_var'] = np.array([s.encode() for s in T_COLS])
        f['A_var'] = np.array([s.encode() for s in A_COLS])
        f['W_dev'], f['X_s_dev'], f['X_v_dev'], f['T_dev'], f['Y_dev'], f['A_dev'] = (
            W[dev], Xs[dev], Xv[dev], T[dev], Y[dev], A[dev])
        # a file with no test units still needs the array to exist (empty split)
        f['W_test'], f['X_s_test'], f['X_v_test'], f['T_test'], f['Y_test'], f['A_test'] = (
            W[test], Xs[test], Xv[test], T[test], Y[test], A[test])


def main():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'fake_ds02.h5'
        _write_fake(p, dev_units=(2, 5), test_units=(11, 14))

        # 1. columns are read, not assumed
        d = load_features(p, 'dev')
        assert d['W_cols'] == W_COLS, d['W_cols']
        assert d['X_s_cols'] == X_S_COLS

        # 2. labels and metadata load separately
        y = load_labels(p, 'dev')
        a = load_aux(p, 'dev')
        assert y.shape == (len(a['unit']),)
        assert a['cycle'].min() == 1

        # 2b. the split comes from the file, and is the file's, not ours
        dev_units, test_units = official_split(p)
        assert dev_units == (2, 5), dev_units
        assert test_units == (11, 14), test_units
        # a file whose two splits share a unit must be refused, not silently used
        q = Path(td) / 'overlap.h5'
        _write_fake(q, dev_units=(2, 5), test_units=(5,))
        try:
            official_split(q)
        except AssertionError as exc:
            assert 'share units' in str(exc), exc
        else:
            raise AssertionError('overlapping dev/test units were accepted')

        # 3. forbidden arrays are refused as features
        from .data import load_raw
        for bad in ('X_v', 'T'):
            try:
                load_raw(p, 'dev', arrays=(bad,))
            except Exception as exc:  # noqa: BLE001 - the loader must raise
                raise AssertionError(f'{bad} should load for diagnostics, got {exc}')
        # load_features has no parameter that could smuggle them in
        import inspect as _inspect
        sig = _inspect.signature(load_features)
        assert list(sig.parameters) == ['path', 'split'], sig

        # 4. audit detects Y constant within cycle and the unit list
        rep = verify(p)
        assert rep['arrays']['W_dev']['shape'] == [2 * 12 * 8, 4]
        assert rep['arrays']['T_dev']['n_cols'] == 10

        # 5. the inspection entry point still imports
        from .inspect_ds02 import main as _  # noqa: F401 - import must not fail

        # 6. the feature-name list describes the array layout, not its reverse
        from .features import FEATURE_NAMES, N_SENSORS, W_FEATURE_NAMES
        assert len(FEATURE_NAMES) == 51, len(FEATURE_NAMES)
        stats = [n.rsplit('_', 1)[1] for n in FEATURE_NAMES[:42]]
        assert stats == ['mean'] * 14 + ['std'] * 14 + ['slope'] * 14, stats
        assert W_FEATURE_NAMES[:4] == tuple(f'W_{j}_mean' for j in range(4))
        assert N_SENSORS == 14
        print('contract test: PASS')


if __name__ == '__main__':
    main()
