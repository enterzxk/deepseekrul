"""Stage D interface contract check (spec D7.1).

`test_d_engine.py` proves the four D7.2 properties on the recursion. It does not
check that `engine_batch` actually honours the *documented* contract, and that is
what a reviewer has to be able to trust before any training code touches it.

Every assertion below is a statement copied from D7.1 or D4.6 of
METHOD_SPECIFICATION.md, not a property invented here:

  keys/shapes   -- the documented fields and their documented shapes;
  valid         -- 1 inside the true length, 0 in the padding;
  padding       -- padded tensors are zero, so nothing leaks into the prefix sum;
  cycle_raw     -- cycle index 1..n at position t = cycle-1;
  r             -- uncapped RUL n - t at that position;
  y             -- min(r, rul_cap), i.e. the censored label;
  cut_t         -- the five protocol cut cycles, matching the frozen val windows.

Usage: python docs/paper_b_hc/probes/d_interface_contract.py
"""
from pathlib import Path

import numpy as np

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.hc.data_hc import build

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul')
DATA = str(ROOT / 'data/CMAPSS')
SPAN, CUTS = C.SPAN, C.VAL_CUTS

KEYS_COMMON = ['x', 'mask', 'baseline', 'cond', 'valid', 'cycle_raw', 'r', 'y',
               'engine_id', 'lengths', 'Lmax', 'split']


def check(subset, split):
    eb = engine_batch(subset, DATA, SPAN, CUTS, split=split)
    E, Lmax = len(eb['lengths']), eb['Lmax']
    n_sensor = eb['x'].shape[3]

    for k in KEYS_COMMON + (['cut_t', 'cut_valid'] if split == 'val' else []):
        assert k in eb, '%s: missing key %s' % (subset, k)
    assert eb['x'].shape == (E, Lmax, SPAN, n_sensor), eb['x'].shape
    assert eb['mask'].shape == (E, Lmax, SPAN), eb['mask'].shape
    assert eb['baseline'].shape == (E, Lmax, n_sensor), eb['baseline'].shape
    assert eb['cond'].shape == (E, Lmax), eb['cond'].shape
    for k in ('valid', 'cycle_raw', 'r', 'y'):
        assert eb[k].shape == (E, Lmax), (k, eb[k].shape)
    assert eb['engine_id'].shape == (E,) and eb['lengths'].shape == (E,)

    for i, n in enumerate(eb['lengths']):
        n = int(n)
        assert eb['valid'][i, :n].min() == 1.0 and eb['valid'][i, :n].max() == 1.0
        assert eb['valid'][i, n:].max(initial=0) == 0.0
        # padding must be exactly zero: a non-zero pad would enter the prefix sum
        # padding must be exactly zero ELEMENT-WISE: a max() check misses an
        # all-negative non-zero pad, which would enter the prefix sum silently.
        assert np.all(eb['x'][i, n:] == 0.0), 'x padding not element-wise zero'
        assert np.all(eb['mask'][i, n:] == 0.0), 'mask padding not element-wise zero'
        assert np.all(eb['baseline'][i, n:] == 0.0), 'baseline padding not element-wise zero'
        assert eb['cycle_raw'][i, :n].tolist() == list(range(1, n + 1)), 'cycle axis'
        assert np.allclose(eb['r'][i, :n], n - np.arange(1, n + 1)), 'r = n - t'
        assert np.allclose(eb['y'][i, :n], np.minimum(eb['r'][i, :n], C.RUL_CAP)), 'y = min(r, cap)'

    n_val_win = 0
    if split == 'val':
        ds = build(subset, DATA, SPAN, CUTS, cycle_source='raw')
        va = ds['val']
        n_val_win = len(va['y'])
        pos = {int(u): i for i, u in enumerate(eb['engine_id'])}
        for j in range(n_val_win):
            uid, cut = int(va['unit'][j]), int(va['cut'][j])
            i = pos[uid]
            row = [int(c) for c, v in zip(eb['cut_t'][i], eb['cut_valid'][i]) if v > 0]
            assert cut in row, '%s: cut %d of engine %d not in cut_t %s' % (subset, cut, uid, row)
            assert int(eb['cycle_raw'][i, cut - 1]) == cut, 'cut misaligned with cycle axis'

    return dict(subset=subset, split=split, engines=E, Lmax=Lmax,
                windows=int(eb['valid'].sum()), val_windows=n_val_win)


def main():
    rows = []
    for subset in ('FD001', 'FD004'):
        for split in ('train', 'val'):
            r = check(subset, split)
            rows.append(r)
            print('%-6s %-5s engines=%3d Lmax=%3d windows=%6d val_windows=%s'
                  % (r['subset'], r['split'], r['engines'], r['Lmax'],
                     r['windows'], r['val_windows'] or '-'))
    print('\nCONTRACT OK: %d checks passed (D7.1 keys/shapes, valid mask, zero padding, '
          'cycle axis, r, y, cut_t)' % len(rows))


if __name__ == '__main__':
    main()
