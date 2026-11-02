"""The audit itself: what can be concluded about conditional coverage from n engines.

Three quantities per cell, and they answer three different questions:

  m, c_b     level error and shape error. `m` is what a marginal PICP reports; `c_b` is
             the only part a band table adds. Reporting "band coverage matches the
             nominal level" while |m| is large is reporting c and claiming m.
  lower_b    the one-sided simultaneous lower confidence bound on the worst band. Always
             defined, never "underpowered" -- but its width is the honest answer.
  mde_b      the smallest band-coverage deficit the cell could have detected. If the mde
             exceeds the deficit anyone cares about, the band table carries no information
             about that deficit regardless of what it shows.

The point of reporting all three is that a p-value hides the third. `p = 0.31` reads as
"no problem" whether the mde is 0.03 or 0.44.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (cover_indicator, wilson, binom_p, mde, randomized_pit,
                  naive_pit, pit_uniform, pit_conditional, calibration_slope)

BANDS_125 = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 126)]
BANDS_100 = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]


def cp_lower(k, n, conf):
    """One-sided Clopper-Pearson lower bound."""
    if n == 0:
        return np.nan
    if k == 0:
        return 0.0
    return float(stats.beta.ppf(1 - conf, k, n - k + 1))


def icc_design_effect(e, unit):
    """One-way random-effects ICC and the design effect it implies.

    N-CMAPSS test sets hold 40-100 flight cycles per engine, so the n in a coverage table
    is not n independent Bernoulli trials. Ignoring that inflates every test's apparent
    precision. n_eff = n / (1 + (mbar - 1) * rho) is the standard correction, and it is
    applied before any power statement is made.
    """
    if unit is None:
        return 0.0, 1.0
    unit = np.asarray(unit)
    groups = [e[unit == u] for u in np.unique(unit)]
    groups = [g for g in groups if len(g) > 0]
    G, n = len(groups), len(e)
    if G < 2 or n <= G:
        return 0.0, 1.0
    gm = np.array([g.mean() for g in groups])
    ng = np.array([len(g) for g in groups])
    gm_overall = e.mean()
    msb = float((ng * (gm - gm_overall) ** 2).sum() / (G - 1))
    msw = float(sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - G))
    mbar = float((n - (ng ** 2).sum() / n) / (G - 1))
    if msb + (mbar - 1) * msw <= 0:
        return 0.0, 1.0
    rho = (msb - msw) / (msb + (mbar - 1) * msw)
    rho = float(max(0.0, min(1.0, rho)))
    return rho, float(1.0 + (mbar - 1) * rho)


def analyse(p, y, yhat, bands, level=0.90, alpha=0.05, unit=None):
    e, lo, hi = cover_indicator(p, y, level)
    n = len(y)
    a_hat = float(e.mean())
    m = a_hat - level
    k_tot = int(e.sum())
    rho, deff = icc_design_effect(e, unit)

    nb = len(bands)
    rows = []
    for lo_b, hi_b in bands:
        msk = (y >= lo_b) & (y <= hi_b) if hi_b > y.max() else (y >= lo_b) & (y < hi_b)
        n_b = int(msk.sum())
        if n_b == 0:
            continue
        k_b = int(e[msk].sum())
        wl, wh = wilson(k_b, n_b)
        n_eff = max(2, int(round(n_b / deff)))
        rows.append(dict(
            band=f"[{lo_b},{hi_b})", n=n_b, n_eff=n_eff, k=k_b, picp=k_b / n_b,
            wilson=[wl, wh], c=k_b / n_b - a_hat, total=k_b / n_b - level,
            bias=float((((p[msk] * np.arange(p.shape[1])[None, :]).sum(1)) - y[msk]).mean()),
            mde_vs_marginal=mde(n_eff, a_hat, alpha=alpha / nb),
            mde_vs_nominal=mde(n_eff, level, alpha=alpha / nb),
        ))

    worst = min(rows, key=lambda r: r["picp"]) if rows else None
    # simultaneous one-sided lower bound on the worst band: Bonferroni over bands,
    # Clopper-Pearson within each. Conservative by construction, which is the point --
    # the alternative is a point estimate with no stated resolution.
    conf = 1 - alpha / max(len(rows), 1)
    for r in rows:
        r["cp_lower_sim"] = cp_lower(r["k"], r["n"], conf)
    lb = min(r["cp_lower_sim"] for r in rows) if rows else np.nan

    rng = np.random.default_rng(0)
    u = randomized_pit(p, y, rng)
    pu = pit_uniform(u)
    pc = pit_conditional(u, yhat)
    pn = pit_uniform(naive_pit(p, y))

    return dict(
        n=n, a_hat=a_hat, m=m, k=k_tot, rho=rho, deff=deff,
        n_eff=float(n / deff),
        wilson_marginal=list(wilson(k_tot, n)),
        p_marginal=binom_p(k_tot, n, level),
        bands=rows,
        worst_band=worst["band"] if worst else None,
        worst_picp=worst["picp"] if worst else None,
        worst_c=worst["c"] if worst else None,
        worst_total=worst["total"] if worst else None,
        min_band_n=min((r["n"] for r in rows), default=0),
        min_band_n_eff=min((r["n_eff"] for r in rows), default=0),
        max_mde=max((r["mde_vs_marginal"] or 0) for r in rows) if rows else None,
        lower_bound_worst=lb,
        pit_random=pu, pit_naive=pn,
        pit_cond=pc,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--bands", default="125")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bands = BANDS_125 if a.bands == "125" else BANDS_100
    files = sorted(glob.glob(a.glob))
    res = {}
    hdr = (f"{'cell':12s} {'n':>5s} {'deff':>5s} {'a90':>5s} {'m':>6s} {'worst':>11s} "
           f"{'picp':>5s} {'c':>6s} {'minNe':>5s} {'mde':>5s} {'lowB':>5s} {'cvm_p':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for f in files:
        d = np.load(f)
        key = Path(f).stem
        unit = d["unit"] if "unit" in d.files else None
        r = analyse(d["p_cens"], d["y"], d["yhat"], bands, unit=unit)
        res[key] = r
        print(f"{key:12s} {r['n']:5d} {r['deff']:5.2f} {r['a_hat']:5.3f} {r['m']:+6.3f} "
              f"{r['worst_band']:>11s} {r['worst_picp']:5.2f} {r['worst_c']:+6.3f} "
              f"{r['min_band_n_eff']:5d} {r['max_mde']:5.2f} {r['lower_bound_worst']:5.2f} "
              f"{r['pit_random']['cvm_p']:7.3f}")
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
