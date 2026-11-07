"""Coverage-Winkler frontier for every method, from saved predictions.

Every method is calibrated on the SAME engine split; the only difference is how
the per-sample interval is formed. Reporting the whole frontier makes the
comparison fair: methods with different realised coverage are not comparable on
a single Winkler number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DELTAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, alpha):
    return float(np.mean((hi - lo) + (2.0 / alpha) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def quantile(vals, delta):
    n = vals.size
    lv = min(1.0, (1.0 - delta) * (1.0 + 1.0 / n))
    return float(np.quantile(vals, lv, method="higher"))


def cqr_frontier(y, lo_raw, hi_raw, cal, tst, label):
    out = []
    for dl in DELTAS:
        q = quantile(np.maximum(lo_raw[cal] - y[cal], y[cal] - hi_raw[cal]), dl)
        lo, hi = lo_raw[tst] - q, hi_raw[tst] + q
        cov = float(np.mean((y[tst] >= lo) & (y[tst] <= hi)))
        out.append((label, dl, cov, winkler(y[tst], lo, hi, dl)))
    return out


def scaled_frontier(y, mu, w, cal, tst, label):
    out = []
    for dl in DELTAS:
        q = quantile(np.abs(y[cal] - mu[cal]) / np.maximum(w[cal], 1e-9), dl)
        lo, hi = mu[tst] - q * w[tst], mu[tst] + q * w[tst]
        cov = float(np.mean((y[tst] >= lo) & (y[tst] <= hi)))
        out.append((label, dl, cov, winkler(y[tst], lo, hi, dl)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/punet_e2e")
    args = ap.parse_args()
    root = Path(args.root)

    print("%-6s %-14s %-6s %-9s %-9s" % ("subset", "method", "delta", "coverage", "winkler"))
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        d = root / sub / "f0_s42"
        pc, pm = d / "preds.csv", d / "preds_mcd.csv"
        if not pc.exists():
            print(sub, "missing preds.csv")
            continue
        c = np.genfromtxt(pc, delimiter=",", names=True)
        cal, tst = engine_split(c["unit"])
        y = c["y_true"]
        mu = (c["y_lo"] + c["y_hi"]) / 2.0

        rows = []
        rows += cqr_frontier(y, c["y_lo"], c["y_hi"], cal, tst, "PU-CQR")
        rows += scaled_frontier(y, mu, np.ones_like(mu), cal, tst, "const-width")
        # QR without calibration: single operating point at alpha=0.1
        lo, hi = c["y_lo"][tst], c["y_hi"][tst]
        rows.append(("QR(raw)", 0.10, float(np.mean((y[tst] >= lo) & (y[tst] <= hi))), winkler(y[tst], lo, hi, 0.10)))
        if pm.exists():
            m = np.genfromtxt(pm, delimiter=",", names=True)
            rows += scaled_frontier(m["y_true"], m["mu"], m["sd"], engine_split(m["unit"])[0], engine_split(m["unit"])[1], "MC-Dropout")
        for label, dl, cov, wk in rows:
            print("%-6s %-14s %-6.2f %-9.3f %-9.1f" % (sub, label, dl, cov, wk))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
