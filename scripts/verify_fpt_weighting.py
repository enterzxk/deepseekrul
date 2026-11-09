"""Does a path-approach-weighted conformal score recover efficiency?

Round-4 showed the uniform simultaneous band is valid but conservative
(FPT coverage 0.997 at target 0.90, median width 101 vs 61 for the invalid
pointwise band).  Here we test whether re-allocating band width towards the
region near the failure threshold -- where the first-passage time is most
sensitive -- keeps coverage while shrinking the interval.

Weighted score (still a max over the path, so Proposition 1 still applies):
    s = max_t  w_t * |X_t - Xhat_t| / sigma
    w_t = 1 + kappa * (1 - Xhat_t) / (1 - L)        (grows as Xhat approaches L)
    band: Xhat_t +/- q * sigma / w_t

CPU only, no training, no dependency on any L2 route decision.

Run: python scripts/verify_fpt_weighting.py
"""

from __future__ import annotations

import numpy as np

L = 0.30
BETA = 1.5
ALPHA = 0.10
SIGMA = 0.02
N_CAL = 400
N_TEST = 2000
T_GRID = np.arange(1, 241)
KAPPAS = [0.0, 0.5, 1.0, 2.0, 4.0]


def path(T, t):
    return 1.0 - (1.0 - L) * (t / T) ** BETA


def predict(T_hat, t):
    return 1.0 - (1.0 - L) * (t / T_hat) ** BETA


def weights(yhat, kappa):
    return 1.0 + kappa * (1.0 - yhat) / (1.0 - L)


def fpt_interval(yhat, q, kappa, t):
    w = weights(yhat, kappa)
    lo = yhat - q * SIGMA / w
    hi = yhat + q * SIGMA / w
    lo_idx = np.flatnonzero(lo <= L)
    hi_idx = np.flatnonzero(hi <= L)
    t_lo = float(t[lo_idx[0]]) if lo_idx.size else 0.0
    t_hi = float(t[hi_idx[0]]) if hi_idx.size else float("inf")
    return t_lo, t_hi


def main() -> int:
    rng = np.random.default_rng(0)
    T_cal = rng.uniform(120, 220, N_CAL)
    T_tst = rng.uniform(120, 220, N_TEST)
    e_cal = rng.normal(0, 0.10, N_CAL)
    e_tst = rng.normal(0, 0.10, N_TEST)

    print(f"{'kappa':>6} {'q':>7} {'coverage':>9} {'median width':>13}")
    rows = []
    for kappa in KAPPAS:
        s_cal = []
        for i in range(N_CAL):
            yhat = predict(T_cal[i] * np.exp(e_cal[i]), T_GRID)
            dev = np.abs(path(T_cal[i], T_GRID) - yhat) * weights(yhat, kappa)
            s_cal.append(np.max(dev) / SIGMA)
        s_cal = np.sort(np.array(s_cal))
        k = int(np.ceil((N_CAL + 1) * (1 - ALPHA))) - 1
        q = float(s_cal[min(k, N_CAL - 1)])

        cov, widths = 0, []
        for i in range(N_TEST):
            T = T_tst[i]
            yhat = predict(T * np.exp(e_tst[i]), T_GRID)
            t_lo, t_hi = fpt_interval(yhat, q, kappa, T_GRID)
            if t_lo <= T <= t_hi:
                cov += 1
            if np.isfinite(t_hi - t_lo):
                widths.append(t_hi - t_lo)
        c = cov / N_TEST
        w = float(np.median(widths)) if widths else float("nan")
        rows.append((kappa, q, c, w))
        print(f"{kappa:>6.1f} {q:>7.2f} {c:>9.3f} {w:>13.1f}")

    # the uniform arm (kappa=0) is the reference; a useful weighting must keep
    # coverage >= 0.88 and cut the median width by at least 10%.
    base = rows[0]
    best = min(rows[1:], key=lambda r: r[3]) if len(rows) > 1 else base
    ok = (base[2] >= 0.88) and (best[2] >= 0.88) and (best[3] <= 0.90 * base[3])
    print(f"\nuniform width {base[3]:.1f} -> best weighted {best[3]:.1f} "
          f"(kappa={best[0]}) at coverage {best[2]:.3f}")
    print("RESULT:", "PASS" if ok else "FAIL",
          "(weighted keeps coverage >= 0.88 and cuts width >= 10%)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
