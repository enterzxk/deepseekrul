"""Add a localized conformal baseline (Lee et al. 2026) to the comparison.

Localized CP weights each calibration residual by similarity to the query point
and takes a weighted quantile, instead of one global quantile. Here the
similarity kernel runs on the predicted RUL, which is available at test time.

Computed from the saved prediction dumps, so no retraining is needed.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

DELTAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, alpha):
    return float(np.mean((hi - lo) + (2.0 / alpha) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def weighted_quantile(vals, weights, level):
    order = np.argsort(vals)
    v, w = vals[order], weights[order]
    cw = np.cumsum(w) / w.sum()
    idx = min(int(np.searchsorted(cw, level)), v.size - 1)
    return float(v[idx])


def localized_interval(y_cal, mu_cal, y_t, mu_t, delta, bandwidth=None):
    resid = np.abs(y_cal - mu_cal)
    if bandwidth is None:
        bandwidth = max(float(np.median(np.abs(mu_cal - np.median(mu_cal)))), 1e-3)
    level = min(1.0, (1 - delta) * (1 + 1 / resid.size))
    lo = np.empty(mu_t.size); hi = np.empty(mu_t.size)
    for i, q in enumerate(mu_t):
        w = np.exp(-np.abs(mu_cal - q) / bandwidth)
        s = weighted_quantile(resid, w, level)
        lo[i], hi[i] = q - s, q + s
    return lo, hi


def main() -> int:
    root = Path("experiments/punet_e2e")
    print("%-6s %-14s %-6s %-9s %-9s" % ("subset", "method", "delta", "coverage", "winkler"))
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        d = root / sub / "f0_s42" / "preds.csv"
        if not d.exists():
            print(sub, "missing"); continue
        c = np.genfromtxt(d, delimiter=",", names=True)
        cal, tst = engine_split(c["unit"])
        y = c["y_true"]
        mu = (c["y_lo"] + c["y_hi"]) / 2.0
        for dl in [0.10, 0.15, 0.20]:
            lo, hi = localized_interval(y[cal], mu[cal], y[tst], mu[tst], dl)
            cov = float(np.mean((y[tst] >= lo) & (y[tst] <= hi)))
            print("%-6s %-14s %-6.2f %-9.3f %-9.1f" % (sub, "LocalizedCP", dl, cov, winkler(y[tst], lo, hi, dl)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
