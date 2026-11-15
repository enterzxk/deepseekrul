"""Training-side high-frequency residual scale for the Noise augmentor.

Scientific meaning (plan section 5.1): the quantity estimated here is a proxy
for the scale of the high-frequency residual of *adjacent observations under
the same operating condition*. It contains measurement noise and may also
contain real fast dynamics and serial correlation. Without an independent
instrument calibration it is NOT the true sensor uncertainty, and results on
simulated noise are not a hardware-noise certification.

Statistics (plan section 5.2-5.3), all on the training engines of the frozen
split only, in the coordinate system the scaler actually outputs:

    d[u,t,j] = x[u,t,j] - x[u,t-1,j]
    sigma[c,j] = median(|d - median(d)|) / (0.67448975 * sqrt(2))

Only pairs with the same engine, cycle difference 1, the same operating
condition cluster, and finite values at both ends enter. No cross-engine
concatenation, no cross-condition differences, no window padding, no repeated
counting through overlapping windows (the full engine sequence is scanned once).

Sufficiency thresholds are frozen before any result is seen:
    every condition: >= 200 adjacent pairs from >= 5 training engines
Fallback: condition -> global (all same-condition adjacent pairs pooled) ->
no noise on that dimension (recorded, never imputed from validation data).

The amplitude cap uses the robust observation scale r[c,j] =
MAD(x[c,j]) / 0.67448975 measured in the same transformed coordinates, so the
cap means "at most 20% of this sensor's robust fluctuation scale" and does not
assume the scaler is a z-score.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MAD_K = 0.67448975
DEFAULT_MIN_PAIRS = 200
DEFAULT_MIN_ENGINES = 5
DEFAULT_MAX_REL = 0.20


def _mad(a, axis=None):
    med = np.median(a, axis=axis, keepdims=True)
    return np.median(np.abs(a - med), axis=axis)


def compute_noise_stats(frame, scaled, labels, n_cond=6,
                        min_pairs=DEFAULT_MIN_PAIRS, min_engines=DEFAULT_MIN_ENGINES):
    """Return the frozen sigma / r matrices plus provenance.

    frame: DataFrame with columns unit, cycle (sorted by unit, cycle)
    scaled: (n_rows, n_sensors) float64, scaler-transformed training values
    labels: (n_rows,) int, operating-condition cluster of every row
    """
    unit = frame['unit'].to_numpy(np.int64)
    cycle = frame['cycle'].to_numpy(np.int64)
    X = np.asarray(scaled, np.float64)
    lab = np.asarray(labels, np.int64)
    n_sensors = X.shape[1]

    step = ((unit[1:] == unit[:-1]) & (cycle[1:] - cycle[:-1] == 1)
            & (lab[1:] == lab[:-1]))
    d = X[1:] - X[:-1]
    finite = (np.isfinite(d).all(axis=1) & np.isfinite(X[1:]).all(axis=1)
              & np.isfinite(X[:-1]).all(axis=1))
    step = step & finite
    lab_next = lab[1:]

    sigma_c = np.zeros((n_cond, n_sensors), np.float64)
    r_c = np.zeros((n_cond, n_sensors), np.float64)
    counts = np.zeros(n_cond, np.int64)
    engines = np.zeros(n_cond, np.int64)
    for c in range(n_cond):
        sel = step & (lab_next == c)
        counts[c] = int(sel.sum())
        engines[c] = int(np.unique(unit[1:][sel]).size) if counts[c] else 0
        if counts[c]:
            sigma_c[c] = _mad(d[sel], axis=0) / (MAD_K * np.sqrt(2.0))
        rows = lab == c
        if rows.any():
            r_c[c] = _mad(X[rows], axis=0) / MAD_K

    gcount = int(step.sum())
    gengine = int(np.unique(unit[1:][step]).size) if gcount else 0
    sigma_g = _mad(d[step], axis=0) / (MAD_K * np.sqrt(2.0)) if gcount else np.zeros(n_sensors)
    r_g = _mad(X, axis=0) / MAD_K

    cond_ok = (counts >= int(min_pairs)) & (engines >= int(min_engines))
    glob_ok = bool(gcount >= int(min_pairs) and gengine >= int(min_engines))

    sigma = np.zeros((n_cond, n_sensors), np.float64)
    r = np.zeros((n_cond, n_sensors), np.float64)
    source = np.empty((n_cond, n_sensors), dtype=object)
    for c in range(n_cond):
        for j in range(n_sensors):
            if cond_ok[c]:
                sigma[c, j] = sigma_c[c, j]
                source[c, j] = 'cond'
            elif glob_ok:
                sigma[c, j] = sigma_g[j]
                source[c, j] = 'global'
            else:
                sigma[c, j] = 0.0
                source[c, j] = 'none'
            if cond_ok[c] and r_c[c, j] > 0:
                r[c, j] = r_c[c, j]
            elif glob_ok and r_g[j] > 0:
                r[c, j] = r_g[j]
            else:
                r[c, j] = 0.0

    out = dict(
        n_sensors=int(n_sensors), n_cond=int(n_cond),
        min_pairs=int(min_pairs), min_engines=int(min_engines), mad_k=MAD_K,
        counts=counts.tolist(), engines=engines.tolist(),
        cond_ok=cond_ok.tolist(),
        global_=dict(counts=gcount, engines=gengine, ok=glob_ok,
                     sigma=sigma_g.tolist(), r=r_g.tolist()),
        sigma_c=sigma_c.tolist(), r_c=r_c.tolist(),
        sigma=sigma.tolist(), r=r.tolist(),
        source=[[str(x) for x in row] for row in source],
        n_zero_sigma=int((sigma <= 0).sum()),
        n_global=int((source == 'global').sum()),
        n_none=int((source == 'none').sum()),
    )
    out['hash'] = stats_hash(sigma, r, min_pairs, min_engines)
    return out


def stats_hash(sigma, r, min_pairs, min_engines):
    h = hashlib.sha256()
    for arr in (np.asarray(sigma, np.float64), np.asarray(r, np.float64)):
        h.update(np.ascontiguousarray(np.round(arr, 12)).tobytes())
    h.update(f'|{min_pairs}|{min_engines}|'.encode())
    return h.hexdigest()[:16]


def effective_scale(stats, alpha, max_rel=DEFAULT_MAX_REL):
    """s = min(alpha * sigma, max_rel * r), with cap bookkeeping."""
    sigma = np.asarray(stats['sigma'], np.float64)
    r = np.asarray(stats['r'], np.float64)
    raw = float(alpha) * sigma
    cap = float(max_rel) * r
    s = np.minimum(raw, cap)
    active = raw > 0
    capped = active & (raw > cap)
    return dict(
        alpha=float(alpha),
        s=s.tolist(),
        cap_fraction=float(capped.sum() / max(1, int(active.sum()))),
        active_dims=int(active.sum()),
        zero_dims=int((s <= 0).sum()),
        s_hash=hashlib.sha256(np.ascontiguousarray(np.round(s, 12)).tobytes()).hexdigest()[:16],
        s_rms=float(np.sqrt(np.mean(s ** 2))),
        sigma_rms=float(np.sqrt(np.mean(sigma ** 2))),
    )


def save_stats(stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=1))
    return path
