"""F2 gate: does the first-passage law under a filtered drift interpolate between
the fixed-drift mixture and the effective-diffusion inverse Gaussian?

Model.  h_tau = h_0 + int_0^tau r_s ds + sigma W_tau,   r_s = exp(rho_s),
        d rho = kappa (mu_rho - rho) dt + sigma_rho dW_rho
so r has mean mu, stationary CV cv, correlation time tau_c = 1/kappa.

Approximations under test (all evaluated against Monte Carlo truth):
  (b) MIXTURE : int IG(t; r, sigma) phi(r; mu, s) dr
                = fixed random drift, the object computed by Si 2013 / Chen & Liu 2022
  (c) EFFDIFF : IG(t; mu, sqrt(sigma^2 + 2 s^2 tau_c))
                = rapid-fluctuation averaging, the claim of Proposition 2
  (d) PLUGIN  : IG(t; mu, sigma)   = drift treated as known exactly (reference floor)

Prop. 2 claims (c) is accurate for tau_c/tau_f << 1, (b) for tau_c/tau_f >> 1, with a
crossover, and that each is measurably wrong in the other's regime.

Run: python fpt_theory_check.py [n_paths] [dt]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr


def simulate_fpt(n_paths, tau_f, tau_c, cv_r, sigma_rel, dt, seed, h0=0.0, H=1.0,
                 t_max_mult=8.0):
    """Monte Carlo first-passage times (in absolute time units)."""
    rng = np.random.default_rng(seed)
    mu = (H - h0) / tau_f
    sigma = sigma_rel * mu
    s_rho = np.sqrt(np.log(1.0 + cv_r ** 2))
    mu_rho = np.log(mu) - 0.5 * s_rho ** 2
    kappa = 1.0 / tau_c
    sigma_rho = s_rho * np.sqrt(2.0 * kappa)

    n_steps = int(np.ceil(t_max_mult * tau_f / dt))
    h = np.full(n_paths, h0)
    rho = rng.normal(mu_rho, s_rho, size=n_paths)
    sqdt = np.sqrt(dt)
    fpt = np.full(n_paths, np.nan)
    alive = np.ones(n_paths, dtype=bool)

    for k in range(n_steps):
        h = h + np.exp(rho) * dt + sigma * sqdt * rng.standard_normal(n_paths)
        rho = rho + kappa * (mu_rho - rho) * dt + sigma_rho * sqdt * rng.standard_normal(n_paths)
        crossed = alive & (h >= H)
        if crossed.any():
            fpt[crossed] = (k + 1) * dt
            alive[crossed] = False
            if not alive.any():
                break
    s = cv_r * mu
    return fpt, dict(mu=mu, sigma=sigma, s=s, sigma_eff=np.sqrt(sigma ** 2 + 2 * s ** 2 * tau_c))


def ig_cdf(t, H, mu, sigma):
    """Inverse-Gaussian first-passage CDF, vectorised, numerically safe."""
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t)
    pos = t > 0
    tt = t[pos]
    st = sigma * np.sqrt(tt)
    a = (mu * tt - H) / st
    b = (-mu * tt - H) / st
    exponent = np.clip(2.0 * mu * H / sigma ** 2, -700, 700)
    out[pos] = np.clip(ndtr(a) + np.exp(exponent) * ndtr(b), 0.0, 1.0)
    return out


def mixture_cdf(t, H, mu_rho, s_rho, sigma, n_grid=401, n_sd=5.0):
    """E_{r}[IG CDF(t; r, sigma)] with r lognormal, i.e. log r ~ N(mu_rho, s_rho^2).

    The stationary law of r = exp(rho) is lognormal, not normal; using a normal
    approximation with matched mean/SD leaves a residual KS error of ~0.05 even in the
    limit where the mixture is exact.
    """
    z = np.linspace(-n_sd, n_sd, n_grid)
    r_grid = np.exp(mu_rho + s_rho * z)
    w = np.exp(-0.5 * z ** 2)
    w /= w.sum()
    cdf = np.zeros_like(np.asarray(t, dtype=np.float64))
    for r, wi in zip(r_grid, w):
        cdf += wi * ig_cdf(t, H, r, sigma)
    return cdf


def main():
    n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    tau_f, cv_r, sigma_rel = 100.0, 0.30, 0.20
    ratios = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    rows = []
    for ratio in ratios:
        tau_c = ratio * tau_f
        t0 = time.time()
        dt_eff = min(dt, 0.05 * tau_c)   # keep the OU rate resolved
        fpt, prm = simulate_fpt(n_paths, tau_f, tau_c, cv_r, sigma_rel, dt_eff,
                                seed=20261101 + int(ratio * 1000))
        cens = float(np.isnan(fpt).mean())
        obs = fpt[~np.isnan(fpt)]
        if len(obs) < 200:
            print(f"ratio={ratio}: too few events ({len(obs)}), censored={cens:.2f}")
            continue

        lo = max(float(np.nanmin(obs)) * 0.5, dt)
        hi = float(np.nanmax(obs))
        t_grid = np.linspace(lo, hi, 400)
        emp = np.searchsorted(np.sort(obs), t_grid, side="right") / len(obs)

        s_rho = np.sqrt(np.log(1.0 + cv_r ** 2))
        mu_rho = np.log(prm["mu"]) - 0.5 * s_rho ** 2
        c_mix = mixture_cdf(t_grid, 1.0, mu_rho, s_rho, prm["sigma"])
        c_eff = ig_cdf(t_grid, 1.0, prm["mu"], prm["sigma_eff"])
        c_plg = ig_cdf(t_grid, 1.0, prm["mu"], prm["sigma"])

        ks = lambda m: float(np.max(np.abs(emp - m)))  # noqa: E731

        # empirical check of the coefficient in sigma_eff^2 = sigma^2 + c*s^2*tau_c
        best_c, best_ks = None, 1e9
        for c in np.linspace(0.0, 4.0, 41):
            se = np.sqrt(prm["sigma"] ** 2 + c * prm["s"] ** 2 * tau_c)
            k = ks(ig_cdf(t_grid, 1.0, prm["mu"], se))
            if k < best_ks:
                best_c, best_ks = float(c), float(k)
        row = dict(ratio=ratio, tau_c=tau_c, censored=round(cens, 4),
                   n_events=int(len(obs)),
                   mean_rel=round(float(obs.mean() / tau_f), 4),
                   sd_rel=round(float(obs.std() / tau_f), 4),
                   sigma_eff_over_sigma=round(float(prm["sigma_eff"] / prm["sigma"]), 2),
                   dt_used=round(dt_eff, 4),
                   ks_mixture=round(ks(c_mix), 4),
                   ks_effdiff=round(ks(c_eff), 4),
                   ks_plugin=round(ks(c_plg), 4),
                   fitted_c=round(best_c, 2), ks_at_fitted=round(best_ks, 4),
                   seconds=round(time.time() - t0, 1))
        row["winner"] = "mixture" if row["ks_mixture"] < row["ks_effdiff"] else "effdiff"
        rows.append(row)
        print(f"tau_c/tau_f={ratio:6.2f} n={row['n_events']:6d} cens={cens:.3f} "
              f"sd={row['sd_rel']:.3f} sig_eff/sig={row['sigma_eff_over_sigma']:6.2f} "
              f"KS mix={row['ks_mixture']:.4f} eff={row['ks_effdiff']:.4f} "
              f"plug={row['ks_plugin']:.4f} c*={row['fitted_c']:.2f}"
              f"(KS {row['ks_at_fitted']:.4f}) -> {row['winner']}", flush=True)

    low = [r for r in rows if r["ratio"] <= 0.03]
    mid = [r for r in rows if 0.03 < r["ratio"] < 10.0]
    high = [r for r in rows if r["ratio"] >= 10.0]
    verdict = dict(
        low_ratio_all_effdiff=all(r["winner"] == "effdiff" for r in low) if low else None,
        high_ratio_all_mixture=all(r["winner"] == "mixture" for r in high) if high else None,
        max_ks_effdiff_low=max([r["ks_effdiff"] for r in low], default=None),
        max_ks_mixture_high=max([r["ks_mixture"] for r in high], default=None),
        fitted_c_low=max([r["fitted_c"] for r in low], default=None),
        fitted_c_low_min=min([r["fitted_c"] for r in low], default=None),
        crossover_ratio=next((r["ratio"] for r in rows if r["winner"] == "mixture"), None),
        mid_band_min_ks=max([min(r["ks_mixture"], r["ks_effdiff"]) for r in mid], default=None),
    )
    gap = [r["ratio"] for r in rows
           if min(r["ks_mixture"], r["ks_effdiff"]) >= 0.05]
    verdict["gap_band"] = [min(gap), max(gap)] if gap else None
    verdict["P2_pass"] = bool(
        verdict["low_ratio_all_effdiff"] and verdict["high_ratio_all_mixture"]
        and (verdict["max_ks_effdiff_low"] or 1) < 0.05
        and (verdict["max_ks_mixture_high"] or 1) < 0.05)
    print("\nVERDICT:", json.dumps(verdict, indent=2))

    path = "/home/yndx/zxk/bearing-rul-pi/results/cido/fpt_theory_check.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(n_paths=n_paths, dt=dt, tau_f=tau_f, cv_r=cv_r,
                   sigma_rel=sigma_rel, rows=rows, verdict=verdict),
              open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
