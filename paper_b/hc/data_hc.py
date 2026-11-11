"""DSPT-HC data layer.

The split, scaler, windowing and HI target are produced by the *frozen*
``paper_b.training.data.prepare_train`` — imported, never re-implemented, so a
protocol drift in the frozen code would show up here immediately instead of
being papered over by a copy.

The one thing this module adds is the **unclipped cycle** vector. The frozen
pipeline stores ``min(cycle/250, 1)``, which throws away the true age for
engines living past 250 cycles — exactly the information the Stage C study
needs. The raw cycle is therefore rebuilt by replaying the same iteration
order as ``make_train_windows`` / ``make_val_windows`` and is *asserted* to
reproduce the frozen clipped vector bit-for-bit. If the two ever disagree the
run aborts rather than training on a misaligned age axis.
"""
import numpy as np

from ..training.data import prepare_train

CYCLE_SCALE = 250.0


def _train_raw_cycle(frame):
    """Same order as make_train_windows: unit sorted, then every cycle."""
    out = []
    for uid, g in frame.groupby('unit', sort=True):
        c = g['cycle'].to_numpy(np.float64)
        # C5: the clipped vector cannot show a break above 250, so check the raw
        # age axis itself. A non-monotone age axis would silently corrupt every
        # cycle claim in Gate A.
        if c.size > 1 and not np.all(np.diff(c) > 0):
            raise AssertionError(f'engine {int(uid)}: raw cycle is not strictly '
                                 f'increasing (min step {np.diff(c).min()})')
        out.append(c)
    return np.concatenate(out)


def _val_raw_cycle(frame, cuts):
    """Same order and de-duplication rule as make_val_windows."""
    out = []
    for uid, g in frame.groupby('unit', sort=True):
        cyc = g['cycle'].to_numpy(np.int64)
        n = len(cyc)
        if n > 1 and not np.all(np.diff(cyc) > 0):
            raise AssertionError(f'engine {int(uid)}: raw cycle is not strictly increasing')
        seen = set()
        for f in cuts:
            c = int(min(n, max(1, round(f * n))))
            if c in seen:
                continue
            seen.add(c)
            out.append(float(cyc[c - 1]))
    return np.asarray(out, np.float64)


def attach_raw_cycle(ds, cuts, span):
    """Return (train_raw, val_raw) in cycles, verified against the frozen clip."""
    tr = _train_raw_cycle(ds['tr'])
    va = _val_raw_cycle(ds['va'], cuts)
    frozen_tr = np.asarray(ds['train']['cycle'], np.float64)
    frozen_va = np.asarray(ds['val']['cycle'], np.float64)
    if tr.shape != frozen_tr.shape:
        raise AssertionError(f'train cycle length {tr.shape} != frozen {frozen_tr.shape}')
    if va.shape != frozen_va.shape:
        raise AssertionError(f'val cycle length {va.shape} != frozen {frozen_va.shape}')
    rec_tr = np.minimum(tr / CYCLE_SCALE, 1.0)
    rec_va = np.minimum(va / CYCLE_SCALE, 1.0)
    if not np.allclose(rec_tr, frozen_tr, atol=1e-6):
        raise AssertionError('rebuilt train cycle does not reproduce the frozen clip')
    if not np.allclose(rec_va, frozen_va, atol=1e-6):
        raise AssertionError('rebuilt val cycle does not reproduce the frozen clip')
    return tr.astype(np.float32), va.astype(np.float32)


def cycle_vector(raw_cycles, source):
    """Map raw cycles to the scalar the encoder consumes (still in cycles)."""
    c = np.asarray(raw_cycles, np.float32)
    if source == 'clipped':
        return np.minimum(c, CYCLE_SCALE).astype(np.float32)
    if source == 'raw':
        return c
    raise KeyError(f'unknown cycle_source {source!r}')


def build(subset, data_dir, span, cuts, cycle_source='raw'):
    """Frozen tensors plus the cycle scalar for this variant."""
    ds = prepare_train(subset, data_dir, span=span, cuts=tuple(cuts))
    tr_raw, va_raw = attach_raw_cycle(ds, cuts, span)
    ds['train']['cycle_hc'] = cycle_vector(tr_raw, cycle_source)
    ds['val']['cycle_hc'] = cycle_vector(va_raw, cycle_source)
    ds['train']['cycle_raw'] = tr_raw
    ds['val']['cycle_raw'] = va_raw
    return ds


def split_fingerprint(ds):
    """Engine-id fingerprint for the manifest (leak check + reproducibility)."""
    import hashlib
    from .config_hc import split_hash
    tr = ','.join(str(int(u)) for u in sorted(ds['train_units']))
    va = ','.join(str(int(u)) for u in sorted(ds['val_units']))
    h = hashlib.sha1(f'{tr}|{va}'.encode()).hexdigest()
    overlap = set(int(u) for u in ds['train_units']) & set(int(u) for u in ds['val_units'])
    return dict(train_n=len(ds['train_units']), val_n=len(ds['val_units']),
                train_ids=tr, val_ids=va, sha1=h,
                val_engine_ids_hash=split_hash(ds['val_units']),
                overlap=sorted(overlap))
