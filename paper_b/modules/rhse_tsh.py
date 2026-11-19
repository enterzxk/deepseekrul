"""RHSE-TSH: studentized-residual SPC monitoring for sensor faults.

Compact implementation of the round-2 design:
  S1 = |r| (Shewhart, bias)
  S2 = two-sided CUSUM of r (drift)
  S3h = EWMA of innovation variance (noise)
  S3l = rolling std of dy over W valid cycles (stuck)
Thresholds are calibrated on clean training engines only; a 2-of-3 valid-cycle
confirmation plus causal latch reduces single-step false alarms.
"""
import numpy as np


def stats(r, dy, mask, k=0.5, lam=0.8, W=12):
    L, N = r.shape
    S1 = np.zeros((L, N), np.float32)
    S2 = np.zeros((L, N), np.float32)
    S3h = np.zeros((L, N), np.float32)
    S3l = np.full((L, N), np.nan, np.float32)
    cp = np.zeros(N, np.float32)
    cn = np.zeros(N, np.float32)
    v = np.ones(N, np.float32)
    prev = None
    hist = []
    for t in range(L):
        if not mask[t]:
            continue
        x = r[t]
        cp = np.maximum(0.0, cp + x - k)
        cn = np.maximum(0.0, cn - x - k)
        S1[t] = np.abs(x)
        S2[t] = np.maximum(cp, cn)
        if prev is not None:
            e2 = 0.5 * (x - prev) ** 2
            v = lam * v + (1.0 - lam) * e2
        prev = x
        S3h[t] = v
        hist.append(dy[t])
        hist = hist[-W:]
        if len(hist) >= W:
            S3l[t] = np.std(np.stack(hist, 0), axis=0)
    return S1, S2, S3h, S3l


def calibrate(tr, sd, rhse, W=12):
    vals = [[], [], [], []]
    for i in range(len(tr['lengths'])):
        n = int(tr['lengths'][i])
        for f in (0.7, 0.85, 0.95):
            t = int(round(f * n)) - 1
            mask = tr['mask'][i, t].astype(bool)
            if mask.sum() < 15:
                continue
            dy = (tr['x'][i, t] - tr['baseline'][i, t]) / sd
            with __import__('torch').no_grad():
                _, _, r = rhse(__import__('torch').tensor(dy[None], dtype=__import__('torch').float32),
                               __import__('torch').tensor(mask[None], dtype=__import__('torch').float32))
            r = r[0].numpy()
            S1, S2, S3h, S3l = stats(r, dy, mask, W=W)
            vals[0].append(S1[mask])
            vals[1].append(S2[mask])
            vals[2].append(S3h[mask])
            good = mask & ~np.isnan(S3l).any(1)
            if good.any():
                vals[3].append(S3l[good])
    q1 = np.percentile(np.concatenate(vals[0], 0), 99.75, axis=0)
    q2 = np.percentile(np.concatenate(vals[1], 0), 99.75, axis=0)
    q3h = np.percentile(np.concatenate(vals[2], 0), 99.75, axis=0)
    q3l = np.percentile(np.concatenate(vals[3], 0), 0.25, axis=0)
    q3l = np.maximum(q3l, 1e-7)
    return dict(q1=q1, q2=q2, q3h=q3h, q3l=q3l)


def detect(r, dy, mask, q, gamma=1.0, W=12, kmax=4):
    S1, S2, S3h, S3l = stats(r, dy, mask, W=W)
    u = np.maximum.reduce([
        S1 / q['q1'][None, :],
        S2 / q['q2'][None, :],
        S3h / q['q3h'][None, :],
        np.where(S3l <= q['q3l'][None, :], 2.0, 0.0),
    ])
    raw = (u > gamma) & mask[:, None]
    flag = np.zeros_like(raw)
    last = []
    for t in range(len(mask)):
        if not mask[t]:
            continue
        last.append(raw[t])
        last = last[-3:]
        confirm = len(last) >= 2 and np.sum(np.stack(last, 0), axis=0) >= 2
        prev = flag[t - 1] if t > 0 else np.zeros(raw.shape[1], bool)
        flag[t] = (confirm | prev) & mask[t]
        if flag[t].sum() > kmax:
            keep = np.argsort(-u[t])[:kmax]
            f = np.zeros(raw.shape[1], bool)
            f[keep] = True
            flag[t] = f
    return flag, u
