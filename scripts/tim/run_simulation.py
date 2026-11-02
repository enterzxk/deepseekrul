"""Simulation: does the construction hit nominal coverage where exchangeability holds?

The real benchmark cannot answer this question -- with 93-235 engines the binomial
standard error at the 90% level is 0.02-0.03, and the calibration/deployment populations
differ anyway. Here the truth is known, so the constructions can be compared on the one
property they claim: coverage at the nominal level, at a width that is not wastefully
large.

Data-generating process (deliberately RUL-shaped):
  u ~ U(0,1) is the normalised age, Y = round(125 * Beta(a(u), b(u))) clipped to 125,
  with a(u) = 2 + 3u, b(u) = 5 - 3u, so the law is right-skewed early in life and
  left-skewed near failure, and the support is bounded by 125 -- the same qualitative
  structure that makes the C-MAPSS residuals asymmetric.
  The predictor is deliberately misspecified: yhat = 125 * clip(0.85u + 0.25u^2 + e, 0, 1)
  with e ~ N(0, 0.10^2).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C

OUT = C.ROOT / "experiments" / "tim"
OUT.mkdir(parents=True, exist_ok=True)
LEVELS = (0.90, 0.50)
N_CAL = (50, 100, 200, 500, 2000)
N_TEST = 4000
REPS = 200


def draw(rng, n):
    """RUL-shaped: a degradation path, observation noise, and a cap at 125.

    The cap is the source of the asymmetry, exactly as in C-MAPSS: a true RUL cannot
    exceed 125, so a model that over-predicts early in life gets a large negative
    residual while the positive side is bounded by 125-yhat.
    """
    u = rng.random(n)
    truth = 125.0 * (1.0 - u)
    y = np.clip(np.round(truth + rng.normal(0, 9.0, n)), 0, 125)
    yhat = 125.0 * (1.0 - u) * 1.06 + rng.normal(0, 5.5, n) + 4.0
    return y, yhat


def gamma_star(r, alpha, grid=201):
    gs = np.linspace(0, alpha, grid)
    w = [np.quantile(r, min(1 - alpha + g, 1.0)) - np.quantile(r, g) for g in gs]
    return float(gs[int(np.argmin(w))])


def main():
    rng = np.random.default_rng(20261102)
    rows = {}
    for n_cal in N_CAL:
        for lv in LEVELS:
            alpha = 1 - lv
            a2 = alpha / 2
            acc = {k: [] for k in ("SC-sym", "SC-asym", "EAC-sym", "EAC-eq", "EAC")}
            wid = {k: [] for k in acc}
            for _ in range(REPS):
                yc, pc = draw(rng, n_cal)
                yt, pt = draw(rng, N_TEST)
                r = yc - pc
                for tag, (slo, shi, al) in {
                    "SC-sym": (-np.abs(r), np.abs(r), alpha),
                    "SC-asym": (r, r, a2),
                }.items():
                    lo, hi = C.split_conformal(pt, slo, shi, al, al)
                    acc[tag].append(C.coverage(lo, hi, yt))
                    wid[tag].append(C.width(lo, hi))
                g = gamma_star(r, alpha)
                for tag, (lo_, hi_) in {
                    "EAC-sym": C.split_sym(pt, r, lv),
                    "EAC-eq": C.split_conformal(pt, r, r, a2, a2),
                    "EAC": C.split_conformal(pt, r, r, g, alpha - g),
                }.items():
                    acc[tag].append(C.coverage(lo_, hi_, yt))
                    wid[tag].append(C.width(lo_, hi_))
            for k in acc:
                rows[f"{k}|{n_cal}|{lv:.2f}"] = dict(
                    cov=float(np.mean(acc[k])), cov_se=float(np.std(acc[k], ddof=1) / np.sqrt(REPS)),
                    width=float(np.mean(wid[k])), width_sd=float(np.std(wid[k], ddof=1)))
            print(f"n_cal={n_cal:5d} lv={lv:.2f}  " + "  ".join(
                f"{k} {np.mean(acc[k]):.3f}/{np.mean(wid[k]):5.1f}" for k in acc))
    (OUT / "simulation.json").write_text(json.dumps(rows, indent=1))
    print("wrote", OUT / "simulation.json")


if __name__ == "__main__":
    main()
