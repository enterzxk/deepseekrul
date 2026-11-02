"""Paired bootstrap over test engines for the two claims the width table makes.

1. sym > asym/sig in width, at the same nominal level, on every subset.
2. base's FD004 shortfall (0.842 at nominal 0.90) is larger than sampling noise, and
   asym/sig closes it.

Engines are resampled with replacement; each replicate recomputes both endpoints from the
stored per-seed construction, so the interval is over the same randomness the point
estimate is exposed to.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts" / "cal"))
from npforward import forward as np_forward, load_weights  # noqa: E402

MAX_RUL, K = 125.0, 126
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SEED, LEVEL, NB = 2027, 0.90, 4000


def read_csv(p):
    us, ys, ps = [], [], []
    for r in csv.DictReader(open(p)):
        us.append(int(r.get("unit_id", r.get("unit"))))
        ys.append(float(r["y_true"]))
        ps.append(float(r["y_pred"]))
    o = np.argsort(us)
    return np.array(us)[o], np.array(ys)[o], np.array(ps)[o]


def load_set(root):
    out = {}
    for f in sorted(root.glob("A_s*_f*.csv")):
        a, b = f.stem.split("_")[1], f.stem.split("_")[2]
        out[(int(a[1:]), int(b[1:]))] = read_csv(f)
    return out


def main():
    w = load_weights(ROOT / "experiments" / "cal" / "calibrators" / "calibrator_A.json")
    rng = np.random.default_rng(2026)
    out = {}
    for s in SUBSETS:
        dev = load_set(ROOT / "experiments" / "cal" / "dev_preds" / s)
        tst = load_set(ROOT / "experiments" / "calx" / "fold_test" / s)
        d = np.load(ROOT / "experiments" / "cal" / "audit" / f"{s}_A.npz")
        y = d["y"].astype(float)
        dy, dh = [], []
        for f in range(5):
            _, yy, pp = dev[(SEED, f)]
            dy.append(yy); dh.append(pp)
        dy, dh = np.concatenate(dy), np.concatenate(dh)
        sig_d = np_forward(w, dh, MAX_RUL)[1]
        r = dy - dh
        mu_k = np.stack([tst[(SEED, f)][2] for f in range(5)], axis=1)
        mu = mu_k.mean(1)
        sig_t = np_forward(w, mu, MAX_RUL)[1]

        al = 1 - LEVEL
        lo_a = np.clip(mu + np.quantile(r / sig_d, al / 2, method="lower") * sig_t, 0, MAX_RUL)
        hi_a = np.clip(mu + np.quantile(r / sig_d, 1 - al / 2, method="higher") * sig_t, 0, MAX_RUL)
        q = np.quantile(np.abs(r), 1 - al / 2, method="higher")
        lo_s = np.clip(mu - q, 0, MAX_RUL)
        hi_s = np.clip(mu + q, 0, MAX_RUL)

        p_base = d["p_cens"]
        cdf = np.cumsum(p_base, axis=1)
        lo_b = np.array([int(np.searchsorted(cdf[i], al / 2)) for i in range(len(y))])
        hi_b = np.array([min(int(np.searchsorted(cdf[i], 1 - al / 2)), K - 1) for i in range(len(y))])

        n = len(y)
        dw, cb, ca, cs = [], [], [], []
        for _ in range(NB):
            i = rng.integers(0, n, n)
            dw.append(np.mean((hi_s - lo_s)[i]) - np.mean((hi_a - lo_a)[i]))
            cb.append(np.mean((lo_b[i] <= y[i]) & (y[i] <= hi_b[i])))
            ca.append(np.mean((lo_a[i] <= y[i]) & (y[i] <= hi_a[i])))
            cs.append(np.mean((lo_s[i] <= y[i]) & (y[i] <= hi_s[i])))
        out[s] = dict(
            n=n,
            width_sym=round(float(np.mean(hi_s - lo_s)), 2),
            width_asym=round(float(np.mean(hi_a - lo_a)), 2),
            width_gain=[round(float(np.percentile(dw, 2.5)), 2), round(float(np.percentile(dw, 97.5)), 2)],
            width_gain_positive=bool(np.percentile(dw, 2.5) > 0),
            cov_base=[round(float(np.percentile(cb, 2.5)), 3), round(float(np.percentile(cb, 97.5)), 3)],
            cov_asym=[round(float(np.percentile(ca, 2.5)), 3), round(float(np.percentile(ca, 97.5)), 3)],
            cov_sym=[round(float(np.percentile(cs, 2.5)), 3), round(float(np.percentile(cs, 97.5)), 3)],
            p_asym_closer=round(float(np.mean(np.abs(np.array(ca) - LEVEL) < np.abs(np.array(cb) - LEVEL))), 3))
        print(s, json.dumps(out[s]), flush=True)

    (ROOT / "experiments" / "calx" / "bootstrap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
