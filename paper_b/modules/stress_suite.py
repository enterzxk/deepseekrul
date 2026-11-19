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


def inject(dy, mask, sensor, fault, mag, rng=None):
    """Return (faulted_dy, positive_mask) for one window.

    dy: (L,N) standardized; mask: (L,) bool.
    positive_mask marks cycles whose injected magnitude is >= 1 sigma (used for
    amplitude-stratified AUC; for drift the ramp is below 1 sigma near t0).
    """
    dy = np.asarray(dy, dtype=np.float64)
    out = dy.copy()
    first, last = valid_span(mask)
    t0 = first + max(1, (last - first) // 2)
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
