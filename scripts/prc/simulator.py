"""Known-truth simulator for protocol-robust comparison (PRC §5.1).

Generates per-engine prediction errors for two models under a set of evaluation
protocols, with a known protocol-robust advantage theta = min_p delta_p.

Model of the data:
  engine effect       u_i ~ N(0, sigma_e)          (shared by both models -> correlated)
  fold effect         v_f ~ N(0, sigma_f)
  seed effect         s_m ~ N(0, sigma_s)          (model-specific)
  protocol bias       b_m(p) : model-specific, protocol-indexed, the estimand
  noise               eps ~ N(0, sigma_noise)

A prediction error is e_{m,i}(p) = b_m(p) + u_i + v_{f(i)} + s_m + eps_{m,i,p}.
delta_p = RMSE_B(p) - RMSE_A(p) is computed on a large "population" draw to give the
ground truth; theta = min_p delta_p. Positive delta = A better.
"""
from __future__ import annotations

import numpy as np


def make_scenario(rng: np.random.Generator, n_engine=100, k_protocol=4,
                  delta_pattern=(0.0, 1.0, 2.0, 0.5), sigma_e=1.0, sigma_f=0.5,
                  sigma_s=0.0, sigma_noise=1.0, n_fold=5, n_seed=3,
                  pop_reps=400_000, seed_noise=True):
    """Return dict with per-engine errors and the ground-truth deltas.

    delta_pattern: per-protocol true *bias* of model B relative to A, in error units.
    The realised delta_p (RMSE difference) is a monotone function of it; the ground
    truth is computed on a large population draw so no closed form is needed.
    """
    k = len(delta_pattern)
    # ground truth on a population draw
    u_pop = rng.normal(0, sigma_e, pop_reps)
    eps_a = rng.normal(0, sigma_noise, (pop_reps, k))
    eps_b = rng.normal(0, sigma_noise, (pop_reps, k))
    delta_true = []
    for p in range(k):
        ea = u_pop + eps_a[:, p]
        eb = u_pop + delta_pattern[p] + eps_b[:, p]
        delta_true.append(float(np.sqrt(np.mean(eb ** 2)) - np.sqrt(np.mean(ea ** 2))))

    # finite sample used by the estimators
    u = rng.normal(0, sigma_e, n_engine)
    fold_of = rng.integers(0, n_fold, n_engine)
    v = rng.normal(0, sigma_f, n_fold)[fold_of]
    err_a = np.zeros((n_engine, k))
    err_b = np.zeros((n_engine, k))
    for p in range(k):
        # seed_noise=False implements CONDITIONAL inference on the trained models:
        # the seed draw is part of the estimator, not of the sampling scheme, so it
        # must not inflate a bootstrap that resamples engines only. Set True for
        # UNCONDITIONAL inference over training randomness, which requires a nested
        # resampling scheme (paper/PRC_METHOD_DESIGN.md section 3.1).
        sa = rng.normal(0, sigma_s, n_seed).mean() if seed_noise else 0.0
        sb = rng.normal(0, sigma_s, n_seed).mean() if seed_noise else 0.0
        err_a[:, p] = u + v + sa + rng.normal(0, sigma_noise, n_engine)
        err_b[:, p] = u + v + sb + delta_pattern[p] + rng.normal(0, sigma_noise, n_engine)

    return dict(err_a=err_a, err_b=err_b, delta_true=np.array(delta_true),
                theta_true=float(min(delta_true)), k=k, n_engine=n_engine,
                pattern=np.array(delta_pattern))
