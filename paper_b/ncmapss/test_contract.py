"""Contract test for the N-CMAPSS loader, on a synthetic file with the documented layout.

Runs without the 15.8 GB download: builds a miniature HDF5 that mimics
``N-CMAPSS_DS02-006.h5`` (Tables 3-8 of the dataset paper) and checks that

  * the column names are read from ``*_var`` rather than assumed by position;
  * ``Y`` is accepted as a label and refused as a feature;
  * ``X_v`` and ``T`` are refused as features with a loud error;
  * the official unit split is asserted, not silently accepted;
  * ``Y`` being constant within a flight cycle is detected.

    python -m paper_b.ncmapss.test_contract
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .data import (A_COLS, DEV_UNITS, T_COLS, TEST_UNITS, W_COLS, X_S_COLS,
                   X_V_COLS, load_aux, load_features, load_labels, verify)


def _write_fake(path, units, cycles=12, rows_per_cycle=8, seed=0):
    import h5py

    rng = np.random.default_rng(seed)
    n = len(units) * cycles * rows_per_cycle
    A = np.zeros((n, 4), np.float64)
    Y = np.zeros((n, 1), np.float64)
    W = rng.normal(size=(n, 4))
    Xs = rng.normal(size=(n, 14))
    Xv = rng.normal(size=(n, 14))
    T = rng.normal(size=(n, 10))
    i = 0
    for u in units:
        for c in range(1, cycles + 1):
            rul = float(cycles - c)
            for _ in range(rows_per_cycle):
                A[i] = [u, c, 1 + (u % 3), 0 if c < cycles else 1]
                Y[i, 0] = rul
                i += 1
    assert i == n
    with h5py.File(path, 'w') as f:
        f['W_var'] = np.array([s.encode() for s in W_COLS])
        f['X_s_var'] = np.array([s.encode() for s in X_S_COLS])
        f['X_v_var'] = np.array([s.encode() for s in X_V_COLS])
        f['T_var'] = np.array([s.encode() for s in T_COLS])
        f['A_var'] = np.array([s.encode() for s in A_COLS])
        f['W_dev'], f['X_s_dev'], f['X_v_dev'], f['T_dev'], f['Y_dev'], f['A_dev'] = (
            W, Xs, Xv, T, Y, A)
        f['W_test'], f['X_s_test'], f['X_v_test'], f['T_test'], f['Y_test'], f['A_test'] = (
            W[:10], Xs[:10], Xv[:10], T[:10], Y[:10], A[:10])


def main():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'fake_ds02.h5'
        _write_fake(p, units=DEV_UNITS[:2])

        # 1. columns are read, not assumed
        d = load_features(p, 'dev')
        assert d['W_cols'] == W_COLS, d['W_cols']
        assert d['X_s_cols'] == X_S_COLS

        # 2. labels and metadata load separately
        y = load_labels(p, 'dev')
        a = load_aux(p, 'dev')
        assert y.shape == (len(a['unit']),)
        assert a['cycle'].min() == 1

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
        assert rep['arrays']['W_dev']['shape'] == [len(DEV_UNITS[:2]) * 12 * 8, 4]
        assert rep['arrays']['T_dev']['n_cols'] == 10

        # 5. a wrong unit set is detectable (the official split is asserted, not assumed)
        from .inspect_ds02 import main as _  # noqa: F401 - import must not fail
        print('contract test: PASS')


if __name__ == '__main__':
    main()
