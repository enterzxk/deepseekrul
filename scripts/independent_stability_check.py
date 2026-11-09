"""Independent check of the stability claim C5 (per-engine coverage is a
stable engine property, not training noise).

Claimed values (CLAIM_LEDGER.md C5, from scripts/engine_coverage_stability.py):
    ICC(1,1) = 0.780 ; cross-seed Spearman = +0.762 ;
    bottom-quartile Jaccard = 0.599 ; within-engine cross-seed SD = 0.032

This script re-derives per-engine coverage from the raw prediction files and
recomputes all four quantities with its own implementation (no imports of the
authoring script, no scipy). The design is stated explicitly because ICC and
Jaccard depend on it:

  unit   = one (subset, fold) cell; the same engines appear under all 4 seeds
           because the calibration/test split is fixed (seed 42) for analysis
  matrix = engines x 4 seeds of per-engine coverage
  ICC(1,1): one-way random effects, single rater (seeds = raters)
  Spearman: rank correlation of per-engine coverage between two seeds
  Jaccard : bottom quartile of engines (by coverage) under two seeds
  SD      : per-engine standard deviation across the 4 seeds

Everything is averaged over the 20 (subset, fold) cells.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DELTA = 0.1
ALPHA = 0.1
SPLIT_SEED = 42
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 7, 2024, 13]


def split(units, frac=0.3, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def per_engine_coverage(d: Path):
    c = np.genfromtxt(d / "preds_cqr.csv", delimiter=",", names=True)
    cal, tst = split(c["unit"])
    y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
    s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
    lv = min(1.0, (1 - DELTA) * (1 + 1 / s.size))
    q = float(np.quantile(s, lv, method="higher"))
    hit = (y >= lo - q) & (y <= hi + q)
    return pd.Series(hit[tst]).groupby(c["unit"][tst]).mean()


def rankdata(v):
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(1, len(v) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    for i, cnt in enumerate(counts):
        if cnt > 1:
            idx = np.where(inv == i)[0]
            ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def icc11(mat):
    """mat: engines x raters."""
    n, k = mat.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = mat.mean()
    row = mat.mean(axis=1)
    msb = k * float(((row - grand) ** 2).sum()) / (n - 1)
    msw = float(((mat - row[:, None]) ** 2).sum()) / (n * (k - 1))
    den = msb + (k - 1) * msw
    return (msb - msw) / den if den > 0 else float("nan")


def main() -> int:
    records = []
    for sub in SUBSETS:
        for fold in range(5):
            for seed in SEEDS:
                d = Path("experiments/punet_matrix") / sub / f"f{fold}_s{seed}"
                if not (d / "preds_cqr.csv").exists():
                    continue
                cov = per_engine_coverage(d)
                for unit, v in cov.items():
                    records.append(dict(subset=sub, fold=fold, seed=seed,
                                        unit=int(unit), cov=float(v)))
    df = pd.DataFrame(records)
    print(f"engine-cell records: {len(df)}")

    iccs, rhos, jacs, sds = [], [], [], []
    for (sub, fold), g in df.groupby(["subset", "fold"]):
        wide = g.pivot_table(index="unit", columns="seed", values="cov")
        wide = wide.dropna()
        if wide.shape[0] < 4:
            continue
        mat = wide[SEEDS].to_numpy(dtype=float)
        iccs.append(icc11(mat))

        pairs = list(itertools.combinations(SEEDS, 2))
        rhos.append(np.nanmean([spearman(wide[a].to_numpy(), wide[b].to_numpy()) for a, b in pairs]))

        k = max(1, int(round(wide.shape[0] * 0.25)))
        jac = []
        for a, b in pairs:
            A = set(wide[a].nsmallest(k).index)
            B = set(wide[b].nsmallest(k).index)
            jac.append(len(A & B) / len(A | B))
        jacs.append(np.mean(jac))

        sds.append(float(np.mean(np.std(mat, axis=1, ddof=1))))

    res = dict(
        n_cells=len(iccs),
        ICC=round(float(np.mean(iccs)), 3),
        ICC_median=round(float(np.median(iccs)), 3),
        spearman=round(float(np.mean(rhos)), 3),
        jaccard_bottom_quartile=round(float(np.mean(jacs)), 3),
        within_engine_sd=round(float(np.mean(sds)), 4),
    )
    print(json.dumps(res, indent=1))

    claimed = dict(ICC=0.780, spearman=0.762, jaccard_bottom_quartile=0.599, within_engine_sd=0.032)
    print()
    fails = []
    for k, v in claimed.items():
        got = res[k]
        tol = 0.01 if k == "within_engine_sd" else 0.02
        ok = abs(got - v) <= tol
        print(f"{'PASS' if ok else 'DIFF'} {k}: got {got} claimed {v}")
        if not ok:
            fails.append(k)
    print(f"\n{len(claimed) - len(fails)}/{len(claimed)} match claimed values")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
