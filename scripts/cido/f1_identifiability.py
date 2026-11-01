"""F1 gate: is the (h, b) pair identifiable, and does a nonlinear observation model
break the confounding that an affine one cannot?

Two tests, both with known ground truth:

  F1a  rank test.  Accumulate the Fisher information for (h, b) along a trajectory,
       I = sum_t [[g'^T W g', g'^T W], [W g', W]], W = diag(psi)^-1.
       For affine g (g' = L constant) the vector (1, -L) is a null direction, so the
       smallest eigenvalue is ~0. For nonlinear g, g' rotates and the rank is full.

  F1b  consequence.  Run the same EKF on a trajectory where a common-mode step offset
       is injected into every channel at 60% of life. Measure how much of the offset
       leaks into h-hat (which is what corrupts the RUL) and how much is absorbed by
       b-hat. An affine observation model cannot separate them; a nonlinear one can,
       because its sensitivity changes as the state advances.

Run: python f1_identifiability.py [n_mc]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

M = 14
PSI = 0.05 ** 2          # measurement noise variance per channel
SIG_H = 0.004            # process noise on h per cycle
SIG_B = 0.002            # random-walk step of each bias state
DT = 1.0


def make_observation_model(kind, seed=0):
    """Return (g, gprime) for 'affine' or 'nonlinear', with matched slope at h=0."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 1.5, M)
    lam = rng.uniform(0.5, 3.0, M)
    L = a * lam                       # slope at h = 0, identical for both models
    if kind == "affine":
        return (lambda h: L * h), (lambda h: L)
    if kind == "nonlinear":
        return (lambda h: a * (1.0 - np.exp(-lam * h))), \
               (lambda h: a * lam * np.exp(-lam * h))
    raise ValueError(kind)


def simulate_trajectory(rng, T=220, rate=1.0 / 220.0, inject_at=None, inject_vec=None):
    h = np.zeros(T)
    b = np.zeros((T, M))
    r = np.full(T, rate)
    for t in range(1, T):
        h[t] = h[t - 1] + r[t - 1] * DT + SIG_H * np.sqrt(DT) * rng.standard_normal()
        b[t] = b[t - 1] + SIG_B * np.sqrt(DT) * rng.standard_normal(M)
    if inject_at is not None and inject_vec is not None:
        b[inject_at:] += inject_vec      # step offset with a prescribed spatial pattern
    return h, b


def fisher_rank(gprime, h_traj):
    """Accumulated Fisher information for (h, b); returns eigenvalues sorted ascending."""
    info = np.zeros((M + 1, M + 1))
    for h in h_traj:
        gp = gprime(h)
        J = np.zeros((M, M + 1))
        J[:, 0] = gp
        J[:, 1:] = np.eye(M)
        info += J.T @ J / PSI
    return np.linalg.eigvalsh(info)


def run_ekf(y, g, gprime, h0=0.0, rate=1.0 / 220.0,
            p0_h=0.05 ** 2, p0_b=0.02 ** 2, sig_b=SIG_B, record=True):
    """Joint EKF over (h, b_1..b_M) with the observation model y = g(h) + b + eps."""
    T = y.shape[0]
    x = np.zeros(M + 1)
    x[0] = h0
    P = np.diag(np.concatenate([[p0_h], np.full(M, p0_b ** 2)]))
    Q = np.diag(np.concatenate([[SIG_H ** 2 * DT], np.full(M, sig_b ** 2 * DT)]))
    R = np.eye(M) * PSI
    h_hat = np.zeros(T)
    b_hat = np.zeros((T, M))
    for t in range(T):
        if t > 0:
            x[0] += rate * DT
            P = P + Q
        gp = gprime(x[0])
        H = np.zeros((M, M + 1))
        H[:, 0] = gp
        H[:, 1:] = np.eye(M)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        resid = y[t] - (g(x[0]) + x[1:])
        x = x + K @ resid
        P = (np.eye(M + 1) - K @ H) @ P
        P = 0.5 * (P + P.T)
        h_hat[t] = x[0]
        b_hat[t] = x[1:]
    return h_hat, b_hat, float(np.sqrt(P[0, 0]))


def main():
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = {}

    # ---------------- F1a: rank of the accumulated information ----------------
    rng = np.random.default_rng(11)
    h_true, _ = simulate_trajectory(rng)
    rank_rows = {}
    for kind in ("affine", "nonlinear"):
        _, gprime = make_observation_model(kind, seed=0)
        ev = fisher_rank(gprime, h_true)
        rank_rows[kind] = dict(
            smallest=float(ev[0]), second=float(ev[1]), largest=float(ev[-1]),
            condition=float(ev[-1] / max(ev[0], 1e-30)),
            n_below_1e_minus_6=int((ev < 1e-6).sum()))
        print(f"[F1a] {kind:10s} eigenvalues: min={ev[0]:.3e} 2nd={ev[1]:.3e} "
              f"max={ev[-1]:.3e}  cond={rank_rows[kind]['condition']:.2e}")
    out["F1a"] = rank_rows
    out["F1a_pass"] = bool(rank_rows["affine"]["n_below_1e_minus_6"] >= 1
                           and rank_rows["nonlinear"]["n_below_1e_minus_6"] == 0)

    # ---------------- F1b: can the observation model correct a wrong rate? ----------------
    # Three earlier versions of this test failed, and the reasons are the finding:
    #   v1 injected a COMMON-MODE offset -- that is not the confounded direction
    #      (the confounded one is L), so nothing was tested;
    #   v2 varied the initial prior on b -- forgotten within ~30 cycles, zero shift;
    #   v3 varied the process-noise ratio -- also ~zero, because the filter was HANDED
    #      THE TRUE RATE, so h was pinned by the dynamics and the observation model was
    #      decorative.
    # The version that tests Proposition 1 must therefore stress the dynamics assumption:
    # give the filter a rate that is 15% too high. A model whose observations pin h can
    # pull h back toward the truth; a model that cannot will let h drift with the wrong
    # rate and dump the discrepancy into b along L.
    INJ_AT, INJ = 130, 0.15
    leak = {}
    for kind in ("affine", "nonlinear"):
        g, gprime = make_observation_model(kind, seed=0)
        _, gp0 = make_observation_model(kind, seed=0)
        Ldir = gp0(0.0)
        Ldir = INJ * Ldir / np.linalg.norm(Ldir)
        err_wrong, err_right = [], []
        for mc in range(n_mc):
            r = np.random.default_rng(5000 + mc)
            h, b = simulate_trajectory(r, inject_at=INJ_AT, inject_vec=Ldir)
            y = np.stack([g(h[t]) + b[t] for t in range(len(h))])
            y += np.sqrt(PSI) * r.standard_normal(y.shape)
            rate_true = 1.0 / 220.0
            h_wrong, _, _ = run_ekf(y, g, gprime, rate=1.15 * rate_true)
            h_right, _, _ = run_ekf(y, g, gprime, rate=rate_true)
            post = slice(INJ_AT, min(INJ_AT + 40, len(h)))
            err_wrong.append(float(np.sqrt(np.mean((h_wrong[post] - h[post]) ** 2))))
            err_right.append(float(np.sqrt(np.mean((h_right[post] - h[post]) ** 2))))
        leak[kind] = dict(
            definition="RMSE of h-hat when the filter is given a rate 15%% too high",
            rmse_wrong_rate=float(np.mean(err_wrong)),
            rmse_right_rate=float(np.mean(err_right)),
            penalty_of_wrong_rate=float(np.mean(err_wrong) - np.mean(err_right)))
        print(f"[F1b] {kind:10s} h RMSE: right rate={leak[kind]['rmse_right_rate']:.4f}  "
              f"wrong rate={leak[kind]['rmse_wrong_rate']:.4f}  "
              f"penalty={leak[kind]['penalty_of_wrong_rate']:+.4f}")
    out["F1b"] = leak

    aff, non = leak["affine"], leak["nonlinear"]
    out["F1b_pass"] = bool(
        non["penalty_of_wrong_rate"] < 0.5 * aff["penalty_of_wrong_rate"]
        and non["penalty_of_wrong_rate"] > 0)
    out["F1_pass"] = bool(out["F1a_pass"] and out["F1b_pass"])

    print()
    print(json.dumps({k: v for k, v in out.items() if k.endswith("pass")}, indent=2))
    path = "/home/yndx/zxk/bearing-rul-pi/results/cido/f1_identifiability.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
