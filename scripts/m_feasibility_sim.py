"""Feasibility study for route M (L2 D034): is there a NEW estimator / sampling
design for engine-level coverage that beats classical two-stage sampling by
>=20% cost at the same accuracy?

Estimand (primary, E3): theta = P(C_i < c), the fraction of engines whose true
per-engine coverage C_i falls below the floor c. This is the quantity our audit
found matters (marginal coverage ~0.88 hides engines at 0.12-0.49).

Sampling model
  N engines; engine i has true coverage C_i ~ F (unknown, heterogeneous);
  we observe m_i windows per engine, each a Bernoulli(C_i) coverage indicator.
  Cost = K per engine run + k per observed window (K >> k).

Classical baseline (strongest simple competitor)
  plug-in:  theta_hat = (1/n) * sum_i 1[ C_hat_i < c ],  C_hat_i = hits_i / m
  with n and m chosen by the classical two-stage optimal allocation for a
  target variance (Cochran), using a moment estimate of the between-engine
  variance of C_i.

Proposed estimator
  empirical-Bayes / model-assisted: fit a Beta(a,b) to the C_i by moments,
  then theta_hat_MA = (1/n) * sum_i P(C_i < c | hits_i, m; a, b) using the
  Beta-binomial posterior. This corrects the threshold misclassification that
  the plug-in suffers when m is small (a noisy C_hat_i near c flips the
  indicator). Allocation is then re-derived from the posterior variance.

Question: at equal cost, does theta_hat_MA reach |error| <= 0.05 at 95%
confidence with >=20% less cost than the plug-in design?

This script reports the cost ratio over a grid of heterogeneity, dependence and
trajectory length. It is a simulation with known truth; the 80-cell data check
is a separate step (the real data cannot provide ground truth).
"""

from __future__ import annotations

import json
import math

import numpy as np

RNG = np.random.default_rng(20261110)
K_COST, W_COST = 100.0, 1.0     # engine run vs window
TARGET_HALF = 0.05              # absolute error target
Z = 1.96


def simulate_truth(n_engines, hetero, rng):
    """True per-engine coverage C_i on (0,1); hetero = sd of logit(C_i)."""
    mu = math.log(0.9 / 0.1)  # centred at 0.9
    z = rng.normal(0.0, hetero, size=n_engines)
    return 1.0 / (1.0 + np.exp(-(mu + z)))


def draw_hits(C, m, rng):
    return rng.binomial(m, C)


def plug_in(hits, m, c):
    return float(np.mean(hits / m < c))


def beta_moments(C_hat, m):
    """Method-of-moments Beta fit to the latent C_i, de-noised for binomial noise."""
    p = float(np.mean(C_hat))
    v_obs = float(np.var(C_hat, ddof=1)) if len(C_hat) > 1 else 0.0
    # E[Var(C_hat|C)] ~ p(1-p)/m ; subtract it to get between-engine variance
    v_between = max(v_obs - p * (1 - p) / max(m, 1), 1e-6)
    k = p * (1 - p) / v_between - 1.0
    k = max(k, 1e-3)
    return p * k, (1 - p) * k  # a, b


def theta_ma(hits, m, c):
    a, b = beta_moments(hits / m, m)
    # P(C < c | hits_i, m) under Beta-binomial posterior
    vals = []
    for h in hits:
        # posterior Beta(a + h, b + m - h); P(C < c) via regularized incomplete beta
        aa, bb = a + h, b + m - h
        vals.append(betainc(aa, bb, c))
    return float(np.mean(vals))


def betainc(a, b, x):
    """Regularized incomplete beta via continued fraction (Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    # continued fraction
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 200):
        m2 = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m2 * (b - m2) * x) / ((a + 2 * m2 - 1) * (a + 2 * m2))
        else:
            num = -((a + m2) * (a + b + m2) * x) / ((a + 2 * m2) * (a + 2 * m2 + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * f / a


def run_design(estimator, C_pop, c, n, m, rng, reps=400):
    """Monte-Carlo RMSE of an estimator at (n engines, m windows)."""
    errs = []
    for _ in range(reps):
        idx = rng.integers(0, len(C_pop), size=n)
        C = C_pop[idx]
        hits = draw_hits(C, m, rng)
        theta = float(np.mean(C < c))
        est = plug_in(hits, m, c) if estimator == "plug" else theta_ma(hits, m, c)
        errs.append(est - theta)
    return float(np.sqrt(np.mean(np.square(errs))))


def main() -> int:
    c = 0.80
    out = []
    for hetero in [0.3, 0.6, 1.0]:
        for n_pop in [200]:
            C_pop = simulate_truth(n_pop, hetero, RNG)
            theta_true = float(np.mean(C_pop < c))
            for m in [8, 32, 128]:
                # find n that gives RMSE <= TARGET_HALF for each estimator
                res = {}
                for est in ["plug", "ma"]:
                    n_need = None
                    for n in [10, 20, 30, 40, 60, 80, 120, 160]:
                        if n > n_pop:
                            break
                        r = run_design(est, C_pop, c, n, m, RNG, reps=300)
                        if r <= TARGET_HALF:
                            n_need = n
                            break
                    res[est] = n_need
                if res["plug"] and res["ma"]:
                    cost_p = res["plug"] * (K_COST + W_COST * m)
                    cost_m = res["ma"] * (K_COST + W_COST * m)
                    ratio = cost_m / cost_p
                    out.append(dict(hetero=hetero, m=m, theta=round(theta_true, 3),
                                    n_plug=res["plug"], n_ma=res["ma"],
                                    cost_ratio=round(ratio, 3)))
                    print(f"hetero={hetero} m={m:3d} theta={theta_true:.3f} "
                          f"n_plug={res['plug']:3d} n_ma={res['ma']:3d} cost_ratio={ratio:.2f}")
                else:
                    print(f"hetero={hetero} m={m:3d} not reached within engine budget "
                          f"(plug={res['plug']}, ma={res['ma']})")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
