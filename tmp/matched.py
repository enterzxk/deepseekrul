"""Winkler at matched coverage (~0.90) by interpolation along each frontier."""

import numpy as np
from pathlib import Path

DELTAS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35]


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, alpha):
    return float(np.mean((hi - lo) + (2.0 / alpha) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def q_at(vals, delta):
    n = vals.size
    return float(np.quantile(vals, min(1.0, (1.0 - delta) * (1.0 + 1.0 / n)), method="higher"))


def frontier_cqr(y, lo, hi, cal, tst):
    pts = []
    for dl in DELTAS:
        q = q_at(np.maximum(lo[cal] - y[cal], y[cal] - hi[cal]), dl)
        L, H = lo[tst] - q, hi[tst] + q
        pts.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, dl)))
    return sorted(pts)


def frontier_scaled(y, mu, w, cal, tst):
    pts = []
    for dl in DELTAS:
        q = q_at(np.abs(y[cal] - mu[cal]) / np.maximum(w[cal], 1e-9), dl)
        L, H = mu[tst] - q * w[tst], mu[tst] + q * w[tst]
        pts.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, dl)))
    return sorted(pts)


def weighted_quantile(vals, weights, level):
    order = np.argsort(vals)
    v, w = vals[order], weights[order]
    cw = np.cumsum(w) / w.sum()
    return float(v[min(int(np.searchsorted(cw, level)), v.size - 1)])


def frontier_localized(y, mu, cal, tst, deltas=None):
    deltas = deltas or [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    resid = np.abs(y[cal] - mu[cal])
    bw = max(float(np.median(np.abs(mu[cal] - np.median(mu[cal])))), 1e-3)
    pts = []
    for dl in deltas:
        lv = min(1.0, (1 - dl) * (1 + 1 / resid.size))
        lo = np.empty(tst.sum()); hi = np.empty(tst.sum())
        for i, q in enumerate(mu[tst]):
            w = np.exp(-np.abs(mu[cal] - q) / bw)
            z = weighted_quantile(resid, w, lv)
            lo[i], hi[i] = q - z, q + z
        pts.append((float(np.mean((y[tst] >= lo) & (y[tst] <= hi))), winkler(y[tst], lo, hi, dl)))
    return sorted(pts)


def at_coverage(pts, target=0.90):
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0)
    return float("nan")


root = Path("experiments/punet_e2e")
print("%-6s %-12s %-10s %-10s" % ("subset", "method", "W@cov0.90", "note"))
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    d = root / sub / "f0_s42"
    c = np.genfromtxt(d / "preds.csv", delimiter=",", names=True)
    cal, tst = engine_split(c["unit"])
    y = c["y_true"]
    mu = (c["y_lo"] + c["y_hi"]) / 2.0
    print("%-6s %-12s %-10.1f %-10s" % (sub, "PU-CQR", at_coverage(frontier_cqr(y, c["y_lo"], c["y_hi"], cal, tst)), ""))
    print("%-6s %-12s %-10.1f %-10s" % (sub, "const-width", at_coverage(frontier_scaled(y, mu, np.ones_like(mu), cal, tst)), ""))
    m = d / "preds_mcd.csv"
    if m.exists():
        mm = np.genfromtxt(m, delimiter=",", names=True)
        c2, t2 = engine_split(mm["unit"])
        print("%-6s %-12s %-10.1f %-10s" % (sub, "MC-Dropout", at_coverage(frontier_scaled(mm["y_true"], mm["mu"], mm["sd"], c2, t2)), ""))
    print("%-6s %-12s %-10.1f %-10s" % (sub, "LocalizedCP", at_coverage(frontier_localized(y, mu, cal, tst)), ""))
    L, H = c["y_lo"][tst], c["y_hi"][tst]
    print("%-6s %-12s %-10s %-10s" % (sub, "QR(raw)", "-", "cov=%.3f" % np.mean((y[tst] >= L) & (y[tst] <= H))))
    print()
