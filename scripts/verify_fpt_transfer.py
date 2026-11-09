"""CPU-only verification of the PS-FPT transfer propositions (candidate C).

Proposition 1: a SIMULTANEOUS path band implies first-passage-time (FPT) coverage.
Proposition 2: pointwise-marginal bands do NOT imply FPT coverage.

This is a synthetic check of the mathematics, not an experiment on C-MAPSS:
no training, no GPU, and it does not depend on any L2 route decision.

Model used for the check
  true path      X_t = 1 - (1-L) * (t/T)^beta,  X_T = L  (so the FPT is exactly T)
  prediction     Xhat_t = 1 - (1-L) * (t/That)^beta,  That = T * exp(eps)
  score          s = max_t |X_t - Xhat_t| / sigma
  band           Xhat_t +/- q * sigma
  derived FPT    T_lo = inf{t: u_t <= L},  T_hi = inf{t: l_t <= L}

Run: python scripts/verify_fpt_transfer.py
Exit 0 iff the simultaneous arm covers ~1-alpha and beats the marginal arm.
"""

from __future__ import annotations

import numpy as np

L = 0.30          # failure threshold on the health index
BETA = 1.5
ALPHA = 0.10
SIGMA = 0.02      # scale used by the score
N_CAL = 400
N_TEST = 2000
T_GRID = np.arange(1, 241)


def path(T: float, t: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - L) * (t / T) ** BETA


def predict(T_hat: float, t: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - L) * (t / T_hat) ** BETA


def fpt_from_band(lower: np.ndarray, upper: np.ndarray, t: np.ndarray):
    """T_lo: first time the LOWER edge reaches L (X >= lower  =>  T <= T(lower) is wrong;
    the correct direction is: lower <= X  =>  lower hits L no later  =>  T >= T(lower)).
    T_hi: first time the UPPER edge reaches L (X <= upper => T <= T(upper)).
    An edge that never crosses yields the trivial bound (0 or +inf).
    """
    lo_idx = np.flatnonzero(lower <= L)
    up_idx = np.flatnonzero(upper <= L)
    t_lo = float(t[lo_idx[0]]) if lo_idx.size else 0.0
    t_hi = float(t[up_idx[0]]) if up_idx.size else float("inf")
    return t_lo, t_hi


def main() -> int:
    rng = np.random.default_rng(0)
    T_cal = rng.uniform(120, 220, N_CAL)
    T_tst = rng.uniform(120, 220, N_TEST)
    eps_cal = rng.normal(0, 0.10, N_CAL)
    eps_tst = rng.normal(0, 0.10, N_TEST)

    # ---- calibration -----------------------------------------------------
    # simultaneous score: max over the whole future path
    s_cal = np.array([np.max(np.abs(path(T_cal[i], T_GRID) - predict(T_cal[i] * np.exp(eps_cal[i]), T_GRID)))
                      for i in range(N_CAL)]) / SIGMA
    k = int(np.ceil((N_CAL + 1) * (1 - ALPHA))) - 1
    q_sim = float(np.sort(s_cal)[min(k, N_CAL - 1)])

    # pointwise-marginal scores: one quantile per time point
    dev_cal = np.array([np.abs(path(T_cal[i], T_GRID) - predict(T_cal[i] * np.exp(eps_cal[i]), T_GRID))
                        for i in range(N_CAL)]) / SIGMA          # (N_CAL, T)
    q_marg = np.quantile(dev_cal, 1 - ALPHA, axis=0)             # (T,)

    # ---- evaluation ------------------------------------------------------
    cov_sim = np.zeros(N_TEST, bool)
    cov_marg = np.zeros(N_TEST, bool)
    width_sim, width_marg = [], []
    for i in range(N_TEST):
        T = T_tst[i]
        T_hat = T * np.exp(eps_tst[i])
        yhat = predict(T_hat, T_GRID)
        true = path(T, T_GRID)

        up = yhat + q_sim * SIGMA
        lo = yhat - q_sim * SIGMA
        t_lo, t_hi = fpt_from_band(lo, up, T_GRID)
        cov_sim[i] = (t_lo <= T <= t_hi)
        width_sim.append(t_hi - t_lo)

        up_m = yhat + q_marg * SIGMA
        lo_m = yhat - q_marg * SIGMA
        t_lo_m, t_hi_m = fpt_from_band(lo_m, up_m, T_GRID)
        cov_marg[i] = (t_lo_m <= T <= t_hi_m)
        width_marg.append(t_hi_m - t_lo_m)

    print(f"target coverage      : {1-ALPHA:.2f}")
    print(f"simultaneous band    : {cov_sim.mean():.3f}   median width {np.median(width_sim):.1f}")
    print(f"pointwise-marginal   : {cov_marg.mean():.3f}   median width {np.median(width_marg):.1f}")
    print(f"q_simultaneous={q_sim:.2f}   mean q_marginal={q_marg.mean():.2f}")

    ok = (cov_sim.mean() >= 0.88) and (cov_marg.mean() < cov_sim.mean() - 0.05)
    print("\nRESULT:", "PASS" if ok else "FAIL",
          "(simultaneous >= 0.88 and marginal at least 5 pp worse)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
