"""Window tensors and the conditional observation model, for the B line.

Everything the models consume goes through here, so that the only thing that
varies between variants is the *representation*, never the data or the split.

The observation model is

    x_t = a(u_t) + b(u_t) * s_t + eps_t

with a, b diagonal (per channel) and fitted on training engines from the mean
degradation curve of each condition, as established by probes B1/B2.  The
inverse is taken in its ridge form

    s_j = (x_j - a_j) * b_j / (b_j^2 + lam)

because a plain division explodes on channels whose gain is small: in FD002 the
per-channel gains span 0.05 to 1.2 across conditions, so 1/b varies by 24x and
the low-gain channels would dominate the aligned signal with amplified noise.
The ridge form is the Wiener solution: it returns the plain inverse when the
gain is large and shrinks towards zero when the channel carries no information
about s at that condition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import data as D
import probe as B1

ROOT = Path(__file__).resolve().parents[1]
GRID = B1.GRID
CAP = 125.0
WINDOW = 30
RIDGE = 1e-2


# --------------------------------------------------------------------------- #
# the conditional affine observation model
# --------------------------------------------------------------------------- #

def pooled_curve(xs, tr, lab, conds, grid=GRID):
    parts, weights = [], []
    for c in conds:
        m, s = B1.condition_curve(xs, tr.rul, lab, tr.unit, tr.engines(), c, grid)
        parts.append(m)
        weights.append(s.astype(float))
    parts, weights = np.stack(parts), np.stack(weights)
    with np.errstate(invalid="ignore"):
        return np.nansum(parts * weights[:, :, None], axis=0) / np.maximum(weights.sum(0), 1.0)[:, None]


def fit_maps(xs, tr, lab, conds, grid=GRID):
    """Per-condition (a, b) against the pooled curve of ``conds``."""
    base = pooled_curve(xs, tr, lab, conds, grid)
    ok = np.isfinite(base).all(1)
    maps = {}
    for c in conds:
        Cc, _ = B1.condition_curve(xs, tr.rul, lab, tr.unit, tr.engines(), c, grid)
        k = ok & np.isfinite(Cc).all(1)
        a, b = B1.fit_diag(Cc[k].T, base[k].T)
        maps[c] = (a, b)
    return maps, base


def interpolate_map(maps: dict, centres: np.ndarray, scale: np.ndarray, target_u: np.ndarray):
    """Linear-in-settings prediction of (a, b) for a setpoint seen only as u."""
    seen = sorted(maps)
    U = centres[seen] / scale
    X = np.column_stack([np.ones(len(seen)), U])
    xt = np.column_stack([np.ones(1), (target_u / scale)[None, :]])
    A = np.stack([maps[c][0] for c in seen])
    B = np.stack([maps[c][1] for c in seen])
    ca, *_ = np.linalg.lstsq(X, A, rcond=None)
    cb, *_ = np.linalg.lstsq(X, B, rcond=None)
    return (xt @ ca)[0], (xt @ cb)[0]


B_MIN = 0.1


def safe_gain(b):
    """Clamp the per-channel gain away from zero, keeping its sign.

    A gain below ``B_MIN`` means the channel barely moves with health at that
    setpoint; dividing by it would amplify noise 20x or more.  Clamping is the
    only option that keeps ``render`` and ``invert`` exact inverses of each
    other, which is what makes the schedule-shift test a certificate rather than
    a numerical accident.
    """
    return np.sign(b + 1e-12) * np.maximum(np.abs(b), B_MIN)


def invert(x, lab, maps):
    """Invert the conditional affine map, row by row."""
    out = np.empty_like(x)
    for c, (a, b) in maps.items():
        m = lab == c
        if m.any():
            out[m] = (x[m] - a) / safe_gain(b)
    return out


def render(x, src, dst_a, dst_b, maps):
    """Replay rows recorded at setpoint ``src`` under a different setpoint."""
    a_s, b_s = maps[src]
    s = (x - a_s) / safe_gain(b_s)
    return dst_a + safe_gain(dst_b) * s


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #

def windows(f: D.Frame, lab: np.ndarray, mu: np.ndarray, sd: np.ndarray,
            L: int = WINDOW, stride: int = 1, last_only: bool = False,
            x_override: np.ndarray | None = None):
    """Sliding windows.  Returns X (n,L,d), U (n,L,3), y (n,), eng (n,)."""
    x = f.x if x_override is None else x_override
    z = (x - mu) / sd
    X, Uu, Y, E = [], [], [], []
    for e in f.engines():
        idx = f.index_of(e)
        idx = idx[np.argsort(f.cycle[idx])]
        n = len(idx)
        if n < L:
            pad = np.repeat(idx[:1], L - n)
            idx = np.concatenate([pad, idx])
            n = L
        starts = [n - L] if last_only else range(0, n - L + 1, stride)
        for s in starts:
            sl = idx[s: s + L]
            X.append(z[sl])
            Uu.append(f.u[sl])
            Y.append(f.rul[idx[s + L - 1]])
            E.append(e)
    return (np.asarray(X, np.float32), np.asarray(Uu, np.float32),
            np.asarray(Y, np.float32), np.asarray(E, np.int32))


def condition_of(u_rows: np.ndarray, centres: np.ndarray, scale: np.ndarray) -> np.ndarray:
    d = ((u_rows[:, None, :] - centres[None]) / scale[None, None, :]) ** 2
    return d.sum(-1).argmin(1)


def fit_one_map(xs, tr, lab, target: int, base_conds, grid=GRID):
    """Fit the (a,b) map of ``target`` against the pooled curve of ``base_conds``.

    Needed to obtain the *true* map of a held-out setpoint (fitted on its own
    rows) so that it can be used to construct training data and to check what an
    interpolated map costs at test time.
    """
    base = pooled_curve(xs, tr, lab, base_conds, grid)
    ok = np.isfinite(base).all(1)
    Cc, _ = B1.condition_curve(xs, tr.rul, lab, tr.unit, tr.engines(), target, grid)
    k = ok & np.isfinite(Cc).all(1)
    return B1.fit_diag(Cc[k].T, base[k].T)
