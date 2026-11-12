"""Stage D data interface: engine-level trajectories (spec D7.1).

Why this is a separate module rather than an addition to ``data_hc``:
``paper_b/hc/data_hc.py`` is one of the fifteen blobs under the v9 code lock
(md5 ``3ad7c111cf0268409bb40d79f7271cee``). Editing it would change the lock and
invalidate every Stage C run that records ``code_md5``. So the Stage D interface
is added *beside* it: ``data_hc`` stays byte-identical, and this module only
imports from it.

Why the frozen window arrays cannot be used directly: they are per-window
samples with no engine or time index, so the damage recursion
``D_t = sum_{u<=t} delta_u`` cannot be evaluated on them. ``engine_batch``
returns full trajectories padded to ``(E, Lmax, ...)`` plus the ``valid`` mask
that keeps padding out of the prefix sum.

``split='val'`` rebuilds the **whole** trajectory of every validation engine,
not just the five cut windows. A cut window at cycle c cannot supply D at c:
the recursion needs every cycle from 1 to c. The frozen cut windows are located
by ``cut_t`` and must be bit-identical to the corresponding trajectory slice --
asserted in ``paper_b/hc/test_d_engine.py::test_two_interface_parity``.
"""
import numpy as np

from ..training.data import make_train_windows
from .config_hc import RUL_CAP, SPAN, VAL_CUTS
from .data_hc import build


def train_lifetime_mean(subset, data_dir, span=SPAN, cuts=VAL_CUTS):
    """Mean run-to-failure length of the TRAINING engines of the frozen split.

    This is the only lifetime statistic the increment head's bias init is
    allowed to use (advisor ruling, 2026-11-13). The training engines run to
    failure, so their true lifetime is observed; validation engines are cut at
    a fraction of theirs and test lifetimes are unknown, so both are excluded.

    Returns the mean plus the full engine list and per-engine lengths, so the
    number can be recomputed by hand from the record instead of trusted.
    """
    ds = build(subset, data_dir, span, cuts, cycle_source='raw')
    sizes = ds['tr'].groupby('unit', sort=True).size().to_numpy(np.int64)
    units = [int(u) for u, _ in ds['tr'].groupby('unit', sort=True)]
    if sizes.size == 0:
        raise RuntimeError('empty training split for %r' % (subset,))
    return dict(mean=float(sizes.mean()), n=int(sizes.size),
                units=units, lengths=[int(x) for x in sizes],
                total=int(sizes.sum()))


def _blocks(win, sizes, tail):
    """Split a flat per-window array into per-engine blocks, padded to Lmax."""
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    lmax = int(np.max(sizes))
    out = np.zeros((len(sizes), lmax) + tail, win.dtype)
    for i in range(len(sizes)):
        blk = win[offs[i]:offs[i + 1]]
        out[i, :len(blk)] = blk
    return out, lmax


def engine_batch(subset, data_dir, span, cuts, split='train'):
    """Engine-level tensors for the Stage D recursion.

    Returns (spec D7.1):

      x, mask, baseline, cond  -- same tensors the frozen encoder consumes, but
                                  indexed by (engine, cycle) instead of window;
      valid                    -- 1 inside the true length, 0 in the padding;
      cycle_raw, r, y          -- true age, uncapped RUL, capped label;
      engine_id, lengths       -- unit id and true length per row;
      cut_t, cut_valid         -- ('val' only) the five protocol cut cycles.

    ``r = n_e - t`` is exact here: validation engines are training engines cut
    at a fraction of their life, so their failure time is known. This is what
    makes the uncapped metric in D4.6 possible.
    """
    ds = build(subset, data_dir, span, cuts, cycle_source='raw')
    if split == 'train':
        frame, win = ds['tr'], ds['train']
    elif split == 'val':
        frame = ds['va']
        # Full trajectory, not the five cuts: see the module docstring.
        win = make_train_windows(frame, ds['va_s'], ds['lv'], ds['pop'], span,
                                 max_rul=RUL_CAP)
    else:
        raise ValueError("split must be 'train' or 'val', got %r" % (split,))

    groups = list(frame.groupby('unit', sort=True))
    sizes = np.asarray([len(g) for _, g in groups], np.int64)
    engine_id = np.asarray([int(u) for u, _ in groups], np.int64)

    x, lmax = _blocks(win['x'], sizes, win['x'].shape[1:])
    mask, _ = _blocks(win['mask'], sizes, win['mask'].shape[1:])
    baseline, _ = _blocks(win['baseline'], sizes, win['baseline'].shape[1:])
    cond, _ = _blocks(win['cond'], sizes, ())
    y, _ = _blocks(win['y'], sizes, ())

    valid = np.zeros((len(sizes), lmax), np.float32)
    r = np.zeros((len(sizes), lmax), np.float32)
    cycle_raw = np.zeros((len(sizes), lmax), np.float32)
    for i, (_, g) in enumerate(groups):
        n = int(sizes[i])
        valid[i, :n] = 1.0
        r[i, :n] = (n - np.arange(1, n + 1)).astype(np.float32)
        cycle_raw[i, :n] = g['cycle'].to_numpy(np.float32)

    out = dict(x=x, mask=mask, baseline=baseline, cond=cond, y=y,
               valid=valid, r=r, cycle_raw=cycle_raw,
               engine_id=engine_id, lengths=sizes, Lmax=lmax, split=split)

    if split == 'val':
        k = len(cuts)
        cut_t = np.zeros((len(sizes), k), np.int64)
        cut_valid = np.zeros((len(sizes), k), np.float32)
        for i in range(len(sizes)):
            n = int(sizes[i])
            seen = []
            for f in cuts:
                c = int(min(n, max(1, round(f * n))))
                if c not in seen:
                    seen.append(c)
            for j, c in enumerate(seen):
                cut_t[i, j] = c
                cut_valid[i, j] = 1.0
        out['cut_t'] = cut_t
        out['cut_valid'] = cut_valid
    return out
