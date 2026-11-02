"""Asymmetric cross-fitted conformal: signed residual quantiles instead of |residual|.

The symmetric rows over-cover by 0.06-0.08 and cost 1.5x width. Diagnosis: the residual
law is strongly right-skewed (the 125 cap and the early-life plateaus put a long arm on the
high side), so the equal-tailed band +/- q_{1-alpha}(|r|) is far wider than the interval
that actually holds 1-alpha mass. A symmetric band is only optimal under symmetry.

This variant keeps everything else identical and drops the symmetry:

    lower = mu_i + q_{alpha/2}(r),   upper = mu_i + q_{1-alpha/2}(r)

with r the out-of-fold signed residuals, optionally normalised by the frozen layer's scale
head. Under exchangeability of the signed scores this is the marginal interval of a
conformal predictive distribution, so it retains the 1-alpha guarantee at every level
simultaneously, and it inherits the skew instead of paying for it.
"""

from __future__ import annotations

import argparse
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
LEVELS = (0.90, 0.50)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SEEDS = (2027, 2028, 2029)


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


def winkler(lo, hi, y, level):
    a = 1 - level
    return float(np.mean((hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)))


def wilson(k, n, z=1.96):
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments" / "calx" / "asym.json"))
    a = ap.parse_args()
    w = load_weights(ROOT / "experiments" / "cal" / "calibrators" / "calibrator_A.json")
    rep = {}

    for s in SUBSETS:
        dev = load_set(ROOT / "experiments" / "cal" / "dev_preds" / s)
        tst = load_set(ROOT / "experiments" / "calx" / "fold_test" / s)
        d = np.load(ROOT / "experiments" / "cal" / "audit" / f"{s}_A.npz")
        y = d["y"].astype(float)
        per = {}
        for seed in SEEDS:
            dy, dh = [], []
            for f in range(5):
                _, yy, pp = dev[(seed, f)]
                dy.append(yy); dh.append(pp)
            dy, dh = np.concatenate(dy), np.concatenate(dh)
            sig_d = np_forward(w, dh, MAX_RUL)[1]
            mu_k = np.stack([tst[(seed, f)][2] for f in range(5)], axis=1)
            mu_bar = mu_k.mean(1)
            sig_t = np_forward(w, mu_bar, MAX_RUL)[1]
            r = dy - dh
            for name, r_use, scale in (("asym", r, np.ones(len(y))),
                                       ("asym/sig", r / sig_d, sig_t)):
                for lv in LEVELS:
                    al = 1 - lv
                    lo_q = float(np.quantile(r_use, al / 2, method="lower"))
                    hi_q = float(np.quantile(r_use, 1 - al / 2, method="higher"))
                    lo = np.clip(mu_bar + lo_q * scale, 0, MAX_RUL)
                    hi = np.clip(mu_bar + hi_q * scale, 0, MAX_RUL)
                    k = int(((lo <= y) & (y <= hi)).sum())
                    per.setdefault(f"{name}@{lv:.2f}", []).append(
                        dict(cov=k / len(y), lo=wilson(k, len(y))[0], hi=wilson(k, len(y))[1],
                             width=float(np.mean(hi - lo)), winkler=winkler(lo, hi, y, lv)))
            # symmetric reference for the same seed, for the skew penalty
            for lv in LEVELS:
                al = 1 - lv
                q = float(np.quantile(np.abs(r), 1 - al / 2, method="higher"))
                lo, hi = np.clip(mu_bar - q, 0, MAX_RUL), np.clip(mu_bar + q, 0, MAX_RUL)
                k = int(((lo <= y) & (y <= hi)).sum())
                per.setdefault(f"sym@{lv:.2f}", []).append(
                    dict(cov=k / len(y), lo=wilson(k, len(y))[0], hi=wilson(k, len(y))[1],
                         width=float(np.mean(hi - lo)), winkler=winkler(lo, hi, y, lv)))

        rep[s] = {k: dict(cov=round(float(np.mean([t["cov"] for t in v])), 4),
                          cov_sd=round(float(np.std([t["cov"] for t in v])), 4),
                          width=round(float(np.mean([t["width"] for t in v])), 2),
                          winkler=round(float(np.mean([t["winkler"] for t in v])), 2))
                  for k, v in per.items()}
        print(s, {k: (v["cov"], v["width"]) for k, v in rep[s].items()}, flush=True)

    Path(a.out).write_text(json.dumps(rep, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
