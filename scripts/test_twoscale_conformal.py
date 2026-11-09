"""Unit test for src/twoscale/conformal.py (numpy only, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.twoscale.conformal import (  # noqa: E402
    conformal_quantile, cqr_scores, intervals, per_engine_coverage, picp, winkler,
)


def main() -> int:
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
        ok = ok and bool(cond)

    n = 2000
    y = rng.normal(0, 1, n)
    lo0 = -np.ones(n)
    hi0 = np.ones(n)

    # scale == 1 must reproduce the frozen CQR procedure exactly
    s = cqr_scores(y, lo0, hi0)
    q = conformal_quantile(s, 0.1)
    check("quantile uses the higher convention",
          abs(q - np.quantile(s, min(1.0, 0.9 * (1 + 1 / n)), method="higher")) < 1e-12)
    lo, hi = intervals(lo0, hi0, q)
    check("coverage near nominal", abs(picp(y, lo, hi) - 0.9) < 0.02, f"{picp(y, lo, hi):.4f}")

    # a per-window scale that matches the heteroscedasticity keeps coverage
    scale = np.abs(rng.normal(0, 1, n)) + 0.5
    y2 = rng.normal(0, 1, n) * scale
    lo0b = -scale
    hi0b = scale
    sb = cqr_scores(y2, lo0b, hi0b)
    qb = conformal_quantile(sb, 0.1)
    lob, hib = intervals(lo0b, hi0b, qb)
    check("heteroscedastic base interval keeps coverage", abs(picp(y2, lob, hib) - 0.9) < 0.02,
          f"{picp(y2, lob, hib):.4f}")

    # an extra per-window scale must also keep coverage when it is the true scale
    sc = np.abs(rng.normal(0, 1, n)) + 0.5
    y3 = rng.normal(0, 1, n) * sc
    lo0c = -np.ones(n)
    hi0c = np.ones(n)
    s3 = cqr_scores(y3, lo0c, hi0c, scale=sc)
    q3 = conformal_quantile(s3, 0.1)
    lo3, hi3 = intervals(lo0c, hi0c, q3, scale=sc)
    check("scaled expansion keeps coverage", abs(picp(y3, lo3, hi3) - 0.9) < 0.03,
          f"{picp(y3, lo3, hi3):.4f}")

    # Winkler must grow when the interval is widened
    w1 = winkler(y, lo, hi)
    lo2, hi2 = intervals(lo0, hi0, q * 2)
    w2 = winkler(y, lo2, hi2)
    check("Winkler grows with width", w2 > w1, f"{w1:.3f} -> {w2:.3f}")

    # per-engine coverage reports the minimum, not the mean
    units = np.repeat(np.arange(20), n // 20)
    cov = per_engine_coverage(y, lo, hi, units)
    check("per-engine dict has min/p05/median/mean",
          set(cov) == {"min", "p05", "median", "mean"}, str(cov))
    check("min <= median", cov["min"] <= cov["median"])

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
