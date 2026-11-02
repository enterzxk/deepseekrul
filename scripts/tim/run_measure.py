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
    # The number is a property of the benchmark and of a stated test, not of any method.
    # The test is written into the JSON so the paper cannot drift from it again.
    assumptions = dict(
        null_coverage=0.90, alpha_family=0.05, bins=len(BANDS),
        correction="bonferroni", power=0.80, sided="two", independence="one query per engine",
        definition="smallest deviation detected reliably (power >= target for every smaller "
                   "alternative), exact two-sided binomial test with the doubling p-value")
    out["detectability_assumptions"] = assumptions
    for s in C.SUBSETS:
        _, _, layer = C.load_fleet(s)
        y = layer["y"].astype(int)
        counts = {f"[{a},{b})": int(np.sum((y >= a) & (y < b))) for a, b in BANDS}
        d = {}
        for k, n in counts.items():
            if n >= 10:
                env, nai = C.mde_detail(n, 0.90, alpha=0.05 / len(BANDS), power=0.80)
                d[k] = dict(n=n, delta_star=env, delta_star_naive=nai)
        out["detectability"][s] = d
    # pooled
    pooled = np.concatenate([C.load_fleet(s)[2]["y"].astype(int) for s in C.SUBSETS])
    d = {}
    for a, b in BANDS:
        n = int(np.sum((pooled >= a) & (pooled < b)))
        env, nai = C.mde_detail(n, 0.90, alpha=0.05 / len(BANDS), power=0.80)
        d[f"[{a},{b})"] = dict(n=n, delta_star=env, delta_star_naive=nai)
    out["detectability"]["pooled"] = d
    # engines needed for a given precision, family-wise 5%, 5 bins.
    # Linear scan, not bisection: delta*(n) is not monotone in n either (n=143 reaches
    # 0.0999 while n=146 is back at 0.1031), so a binary search returns a non-minimal n.
    need = {f"{t:.3f}": None for t in (0.10, 0.075, 0.05, 0.03)}
    targets = sorted((float(k), k) for k in need)
    # One upward pass: delta*(n) is not monotone in n, so the minimum n for a target has
    # to be found by scanning, and a binary search returns a non-minimal answer
    # (n=143 reaches 0.0999 while n=146 is back at 0.1031).
    for n in range(10, 1500):
        d = C.mde(n, 0.90, alpha=0.05 / 5, power=0.80)
        for t, key in targets:
            if need[key] is None and d <= t:
                need[key] = n
        if all(v is not None for v in need.values()):
            break
    out["engines_needed_per_bin"] = need
    print("engines needed per bin:", need)

    # ------------------------------------------- 3. paired test, clustered by engine
    # The engine is the independent unit.  Three seeds predict the same 93-235 engines,
    # so counting (engine, seed) pairs as independent observations multiplies the sample
    # size by three without adding independent information.  Both the naive and the
    # clustered versions are reported so the inflation is visible.
    from run_matrix import best_gamma, fit_scale as _fit_scale  # noqa
    from run_object import aligned

    rng = np.random.default_rng(20261102)
    for s in C.SUBSETS:
        dev, tst, layer = C.load_fleet(s)
        y = layer["y"].astype(float)
        P15 = np.mean([tst[(sd, f)][2] for sd in C.SEEDS for f in range(5)], axis=0)

        # per-seed coverage indicators, aligned on engine (test arrays share unit order)
        cov_a_s, cov_s_s = [], []
        for seed in C.SEEDS:
            rr, pp = [], []
            for f in range(5):
                _units, yc, preds = aligned(dev, C.SEEDS, f)
                p = preds[seed]
                rr.append(yc - p); pp.append(p)
            r = np.concatenate(rr); pc = np.concatenate(pp)
            sigma = _fit_scale(pc, r)
            z = r / sigma(pc)
            sz = sigma(P15)
            g = best_gamma(z, 0.10)
            lo_a, hi_a = C.split_conformal(P15, z, z, g, 0.10 - g, scale_test=sz)
            lo_s, hi_s = C.split_sym(P15, z, 0.90, scale_test=sz)
            cov_a_s.append(((lo_a <= y) & (y <= hi_a)).astype(float))
            cov_s_s.append(((lo_s <= y) & (y <= hi_s)).astype(float))
        cov_a_s = np.stack(cov_a_s)          # (seeds, engines)
        cov_s_s = np.stack(cov_s_s)

        # naive: every (engine, seed) is an observation
        ca = cov_a_s.ravel().astype(bool); cs = cov_s_s.ravel().astype(bool)
        n01 = int(np.sum(cs & ~ca)); n10 = int(np.sum(ca & ~cs))
        p_naive = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if (n01 + n10) else 1.0

        # clustered: average over seeds inside each engine, then test across engines
        d = cov_a_s.mean(0) - cov_s_s.mean(0)
        nz = d[d != 0]
        p_sign = float(stats.binomtest(int(np.sum(nz > 0)), len(nz), 0.5).pvalue) if len(nz) else 1.0
        obs = float(d.mean())
        flips = rng.choice([-1.0, 1.0], size=(10000, len(d)))
        null = np.abs((flips * d).mean(1))
        p_perm = float(np.mean(null >= abs(obs) - 1e-12))
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])

        out["mcnemar"][s] = dict(
            # naive, seeds treated as independent observations
            sym_only=n01, asym_only=n10, p_naive=p_naive, n_obs=int(ca.size),
            # clustered on the engine
            n_engines=int(len(d)), n_engines_discordant=int(len(nz)),
            asym_advantage=obs, p_sign=p_sign, p_permutation=p_perm,
            diff_lo=float(np.percentile(boot, 2.5)), diff_hi=float(np.percentile(boot, 97.5)),
            cov_asym=float(cov_a_s.mean()), cov_sym=float(cov_s_s.mean()))
        print(f"{s} naive p={p_naive:.2e} (n={ca.size}) | clustered p_perm={p_perm:.3f} "
              f"sign={p_sign:.3f}  diff {obs:+.3f} [{np.percentile(boot,2.5):+.3f},{np.percentile(boot,97.5):+.3f}] "
              f"discordant engines {len(nz)}/{len(d)}")

    (OUT / "measure.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "measure.json")


if __name__ == "__main__":
    main()
