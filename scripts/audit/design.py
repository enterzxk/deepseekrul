"""Three design calculations the audit rests on.

E1  Monte-Carlo check of `mde()`. If the closed-form minimum detectable effect is wrong,
    every claim about "the band table cannot see a 0.10 drop" is wrong with it.

E2  Equal-width vs equal-count binning. The 5-band tables everyone prints are equal-width
    in RUL, which puts the fewest engines exactly where the decisions are. Equal-count
    bands cost nothing but a coarser statement, and they buy back the power.

E3  How many engines a band needs. Reported as the n at which a 0.10 deficit is detected
    with 80% power, then compared against what the benchmarks actually contain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import mde, power_at  # noqa: E402


def e1_mde_check(ns=(20, 40, 80, 150, 300), p0=0.90, delta=0.10, alpha=0.05, reps=4000):
    """Empirical rejection rate at p1 = p0 - delta vs the rate `mde` implies."""
    rng = np.random.default_rng(3)
    out = []
    for n in ns:
        p1 = p0 - delta
        k = rng.binomial(n, p1, reps)
        crit_lo = stats.binom.ppf(alpha / 2, n, p0) - 1
        crit_hi = stats.binom.isf(alpha / 2, n, p0) + 1
        emp = float(np.mean((k <= crit_lo) | (k >= crit_hi)))
        out.append(dict(n=n, p1=p1, empirical=emp, analytic=power_at(n, p0, p1, alpha),
                        mde=mde(n, p0, alpha=alpha)))
    return out


def e2_binning(y, bands, n_bands=None):
    """Equal-width (the given band edges) vs equal-count bands on the same labels.

    The band edges are passed in rather than derived from the label range, so that this
    table counts exactly the same cells as the audit tables. Deriving them from
    min/max would silently shift every edge and make the two tables incomparable.
    """
    y = np.asarray(y)
    n_bands = len(bands)
    hi_all = float(y.max())

    def count(bs):
        out = []
        for a, b in bs:
            m = (y >= a) & (y <= b) if b > hi_all else (y >= a) & (y < b)
            out.append(int(m.sum()))
        return out

    ce = count(bands)
    qs = np.quantile(y, np.linspace(0, 1, n_bands + 1))
    ec_edges = [(float(qs[i]), float(qs[i + 1])) for i in range(n_bands)]
    # equal-count edges are closed on the left and open on the right except the last,
    # which must absorb the maximum label
    cc = []
    for i, (a, b) in enumerate(ec_edges):
        last = i == n_bands - 1
        m = (y >= a) & (y <= b + 1e-9) if last else (y >= a) & (y < b)
        cc.append(int(m.sum()))

    return dict(equal_width=ce, equal_count=cc,
                edges_equal_count=[float(q) for q in qs],
                mde_equal_width=[mde(n, 0.90, alpha=0.01) if n else None for n in ce],
                mde_equal_count=[mde(n, 0.90, alpha=0.01) if n else None for n in cc])


def e3_engines_needed(p0=0.90, delta=0.10, alpha=0.05, power=0.80, bands=5):
    need = None
    for n in range(10, 4001):
        m = mde(n, p0, alpha=alpha / bands)
        if m is not None and m <= delta:
            need = n
            break
    per_band_uncorrected = None
    for n in range(10, 4001):
        m = mde(n, p0, alpha=alpha)
        if m is not None and m <= delta:
            per_band_uncorrected = n
            break
    return dict(delta=delta, alpha=alpha, bands=bands, power=power,
                n_per_band_bonferroni=need, n_per_band_uncorrected=per_band_uncorrected,
                engines_for_5_bands_bonferroni=None if need is None else need * bands,
                engines_for_5_bands_uncorrected=(None if per_band_uncorrected is None
                                                 else per_band_uncorrected * bands))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="experiments/cal/audit/FD*_*.npz")
    ap.add_argument("--out", default="experiments/cal/audit/design.json")
    a = ap.parse_args()

    import glob
    # one label per engine: arm A only, so the 3 backbones do not triple-count
    files = [f for f in sorted(glob.glob(a.glob)) if f.endswith("_A.npz")]
    ycat = np.concatenate([np.load(f)["y"] for f in files])
    bands = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 126)]
    res = dict(e1=e1_mde_check(), e2=e2_binning(ycat, bands),
               e2_per_fleet={Path(f).stem: e2_binning(np.load(f)["y"], bands)
                             for f in files},
               e3_5bands=e3_engines_needed(bands=5),
               e3_3bands=e3_engines_needed(bands=3),
               n_labels_total=int(len(ycat)))
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))

    print("E1  closed-form MDE vs Monte-Carlo rejection rate (p0=0.90, delta=0.10)")
    print(f"{'n':>5s} {'empirical':>10s} {'analytic':>9s} {'mde':>6s}")
    for r in res["e1"]:
        print(f"{r['n']:5d} {r['empirical']:10.3f} {r['analytic']:9.3f} {r['mde']:6.3f}")
    print(f"\nE2  band counts, equal-width vs equal-count ({res['n_labels_total']} engines, arm A)")
    print(f"  pooled equal-width : {res['e2']['equal_width']}  "
          f"mde={[None if v is None else round(v,2) for v in res['e2']['mde_equal_width']]}")
    print(f"  pooled equal-count : {res['e2']['equal_count']}  "
          f"mde={[None if v is None else round(v,2) for v in res['e2']['mde_equal_count']]}")
    for k, v in res["e2_per_fleet"].items():
        print(f"  {k:10s} width={v['equal_width']} count={v['equal_count']}")
    print("\nE3  engines needed per band to see a 0.10 deficit at 80% power")
    for k in ("e3_5bands", "e3_3bands"):
        r = res[k]
        print(f"  {k}: bonf n/band={r['n_per_band_bonferroni']} "
              f"-> {r['engines_for_5_bands_bonferroni']} engines | "
              f"uncorr n/band={r['n_per_band_uncorrected']} "
              f"-> {r['engines_for_5_bands_uncorrected']}")


if __name__ == "__main__":
    main()
