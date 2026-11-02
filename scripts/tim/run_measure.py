"""Measurement results: how much of what a coverage table reports is measurement error.

Three quantities, all computable from artefacts already in the repository:

1. the convention bias -- the same predictive distribution, scored with integer endpoints
   versus randomised PIT, differs by a fixed amount that has nothing to do with the model;
2. the detectability limit -- the smallest coverage deviation a test of size n can find,
   which depends only on how many engines fall in the bin, not on the method;
3. paired discordance -- whether two methods differ on the same engines beyond the
   engines where they disagree at all (McNemar).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C

OUT = C.ROOT / "experiments" / "tim"
OUT.mkdir(parents=True, exist_ok=True)
LEVELS = (0.90, 0.50)
BANDS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 126)]


def main():
    out = {"convention": {}, "detectability": {}, "mcnemar": {}}

    # ---------------------------------------------------------------- 1. convention
    for s in C.SUBSETS:
        _, _, layer = C.load_fleet(s)
        y = layer["y"].astype(int)
        row = {}
        for lv in LEVELS:
            lo, hi = C.interval_from_pmf(layer["p_cens"], lv)
            row[f"{lv:.2f}"] = dict(
                randomized=C.randomized_pit_cov(layer["p_cens"], y, lv, nrep=400),
                integer=C.coverage(lo, hi, y.astype(float)),
                gap=float(C.coverage(lo, hi, y.astype(float))
                          - C.randomized_pit_cov(layer["p_cens"], y, lv, nrep=400)))
        out["convention"][s] = row
        print(f"{s} convention: 90% gap {row['0.90']['gap']:+.3f}  50% gap {row['0.50']['gap']:+.3f}")

    # ---------------------------------------------------------------- 2. detectability
    for s in C.SUBSETS:
        _, _, layer = C.load_fleet(s)
        y = layer["y"].astype(int)
        counts = {f"[{a},{b})": int(np.sum((y >= a) & (y < b))) for a, b in BANDS}
        d = {}
        for k, n in counts.items():
            if n >= 10:
                d[k] = dict(n=n, delta_star=float(C.mde(n, 0.90, alpha=0.05 / len(BANDS), power=0.80)))
        out["detectability"][s] = d
    # pooled
    pooled = np.concatenate([C.load_fleet(s)[2]["y"].astype(int) for s in C.SUBSETS])
    d = {}
    for a, b in BANDS:
        n = int(np.sum((pooled >= a) & (pooled < b)))
        d[f"[{a},{b})"] = dict(n=n, delta_star=float(C.mde(n, 0.90, alpha=0.05 / len(BANDS), power=0.80)))
    out["detectability"]["pooled"] = d
    # engines needed for a given precision, family-wise 5%, 5 bins
    need = {}
    for target in (0.10, 0.075, 0.05, 0.03):
        lo, hi = 10, 20000
        while lo < hi:
            mid = (lo + hi) // 2
            if C.mde(mid, 0.90, alpha=0.05 / 5, power=0.80) <= target:
                hi = mid
            else:
                lo = mid + 1
        need[f"{target:.3f}"] = int(lo)
    out["engines_needed_per_bin"] = need
    print("engines needed per bin:", need)

    # ---------------------------------------------------------------- 3. McNemar
    from run_matrix import best_gamma, fit_scale as _fit_scale  # noqa

    for s in C.SUBSETS:
        dev, tst, layer = C.load_fleet(s)
        y = layer["y"].astype(float)
        n01 = n10 = 0
        cov_a = cov_b = 0
        for seed in C.SEEDS:
            mu = np.stack([tst[(seed, f)][2] for f in range(5)], axis=1)
            yt = tst[(seed, 0)][1]
            r, pc = [], []
            for f in range(5):
                yc = dev[(seed, f)][1]
                p = np.mean([dev[(x, f)][2] for x in C.SEEDS], axis=0)
                r.append(yc - p)
                pc.append(p)
            r = np.concatenate(r)
            pc = np.concatenate(pc)
            sigma = _fit_scale(pc, r)
            z = r / sigma(pc)
            pbar = mu.mean(1)
            sz = sigma(pbar)
            g = best_gamma(z, 0.10)
            lo_a, hi_a = C.split_conformal(pbar, z, z, g, 0.10 - g, scale_test=sz)
            lo_s, hi_s = C.split_sym(pbar, z, 0.90, scale_test=sz)
            ca = (lo_a <= yt) & (yt <= hi_a)
            cs = (lo_s <= yt) & (yt <= hi_s)
            n01 += int(np.sum(cs & ~ca))     # covered by symmetric only
            n10 += int(np.sum(ca & ~cs))     # covered by asymmetric only
            cov_a += ca.mean()
            cov_b += cs.mean()
        p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if (n01 + n10) else 1.0
        out["mcnemar"][s] = dict(sym_only=n01, asym_only=n10, p=p,
                                 cov_asym=float(cov_a / 3), cov_sym=float(cov_b / 3))
        print(f"{s} McNemar sym-only {n01} asym-only {n10} p={p:.2e}")

    (OUT / "measure.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "measure.json")


if __name__ == "__main__":
    main()
