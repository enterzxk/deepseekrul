"""Fixed sensor-fault injection protocol for detection / RUL robustness.

All injections are in the standardized dy space (z-score units) unless noted.
`stuck` in z-space is a constant; in raw physical space a stuck sensor is not
constant after condition normalization, which is a known limitation to revisit.
"""
import numpy as np


def valid_span(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0, 0
    return int(idx[0]), int(idx[-1] + 1)


def inject(dy, mask, sensor, fault, mag, rng=None, onset=0.5):
    """Return (faulted_dy, positive_mask) for one window.

    dy: (L,N) standardized; mask: (L,) bool; onset in (0,1) is the fraction of
    the valid history at which the fault starts.
    """
    dy = np.asarray(dy, dtype=np.float64)
    out = dy.copy()
    first, last = valid_span(mask)
    t0 = first + max(1, int(onset * (last - first)))
    pos = np.zeros(len(dy), dtype=bool)
    if fault == 'bias':
        out[t0:last, sensor] += mag
        pos[t0:last] = True
    elif fault == 'drift':
        ramp = np.linspace(0.0, mag, max(1, last - t0))
        out[t0:last, sensor] += ramp
        pos[t0:last] = ramp >= 1.0
    elif fault == 'noise':
        if rng is None:
            rng = np.random.default_rng(0)
        out[t0:last, sensor] += rng.normal(0.0, mag, max(1, last - t0))
        pos[t0:last] = True
    elif fault == 'stuck':
        out[t0:last, sensor] = out[t0, sensor]
        pos[t0:last] = True
    else:
        raise ValueError('unknown fault %r' % (fault,))
    return out, pos
