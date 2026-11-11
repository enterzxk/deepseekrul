"""Autocorrelation length of the degradation signal vs the MS kernel sizes (3/5/7)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_b.training.data import _prepare_split


def acf(x, max_lag):
    x = x - x.mean()
    v = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / v if k else 1.0 for k in range(max_lag + 1)])


for subset in ('FD001', 'FD002', 'FD003', 'FD004'):
    s = _prepare_split(subset, Path('data/CMAPSS'), 7, 0, 0.2, 6)
    Z = s['tr_s']; units = s['tr'].unit.values
    curves = []
    for u in np.unique(units):
        seq = Z[units == u]
        if len(seq) < 60: continue
        _, _, vt = np.linalg.svd(seq - seq.mean(0), full_matrices=False)
        pc = (seq - seq.mean(0)) @ vt[0]
        if np.corrcoef(pc, np.arange(len(pc)))[0, 1] < 0: pc = -pc
        curves.append(acf(pc, 50))
    A = np.mean(curves, axis=0)
    lag50 = int(np.argmax(A < 0.5)) if np.any(A < 0.5) else 50
    lag90 = int(np.argmax(A < 0.9)) if np.any(A < 0.9) else 50
    print(f'{subset}: acf(1)={A[1]:.3f} acf(7)={A[7]:.3f} acf(20)={A[20]:.3f} '
          f'lag(acf<0.9)={lag90} lag(acf<0.5)={lag50}')
