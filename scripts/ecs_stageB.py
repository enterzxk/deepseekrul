"""ECS Stage B: cross-fitted predictive information gain -> credibility factors.

For each of the 5 folds, the Stage-A likelihood model of that fold is applied to the
fold's held-out engines (out-of-sample by construction). Per engine:

  l0 = log p0(y | age)                prior alone
  l1 = log p1(y | window, age)        likelihood alone
  delta = l1 - l0                     realised information gain
  kappa = min(1, age / 50)            coverage of the prediction window
  rho   = cross-channel trend agreement in the last window

H3 is the test that delta increases with kappa and rho (one-sided, alpha = 0.05).
If either slope is <= 0, the two-factor construction is false and the method reduces
to a constant alpha - recorded as a falsification, not patched.

Usage: python ecs_stageB.py --subset FD002 --seed 2027
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np

ROOT = "/home/yndx/zxk/bearing-rul-pi"
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/scripts")

import dsh_e21_test_all as e21
import run_candidate as rc
import torch
from ecs.prior import AgeConditionalPrior
from ecs.likelihood import BinLikelihood, rul_to_bin, log_prob_np, N_BINS
from dataset import SELECTED_SENSOR_COLUMNS

FOLDS, WINDOW = (0, 1, 2, 3, 4), 50
# Multi-cutoff sampling on development engines. The matched protocol used in training
# forces age >= 50, so kappa has no variance there and H3 cannot be tested; sampling
# ages below the window (with the same first-frame padding rule) creates the variation
# that the official test set exhibits naturally.
AGES = (20, 25, 30, 35, 40, 50, 60, 80, 100, 120, 150)


def build_at_ages(frame, values, ages=AGES, win=WINDOW, cap=125.0):
    """Windows at multiple observation ages, padded like the frozen test rule."""
    xs, ys, es, us, ag = [], [], [], [], []
    for u, g in frame.groupby("unit", sort=True):
        seq = np.asarray(values[g.index], dtype=np.float32)
        L = len(seq)
        picks = [a for a in ages if a <= L]
        if not picks:
            picks = [L]
        early = seq[: min(5, L)]
        baseline = early.mean(0)
        ref_std = float(early.std(0).mean())
        for a in picks:
            w = seq[max(0, a - win):a]
            if len(w) < win:
                w = np.concatenate([np.repeat(w[:1], win - len(w), axis=0), w], axis=0)
            dev = float(np.abs(w - baseline).mean())
            ema = seq[0].copy()
            for r in seq[:a]:
                ema = 0.5 * ema + 0.5 * r
            jitter = float(np.abs(seq[a - 1] - ema).mean())
            ev = [1.0, 1.0, float(np.exp(-3.0 / 30.0)), float(np.exp(-dev)),
                  1.0 / (1.0 + ref_std), jitter]
            xs.append(np.ascontiguousarray(w))
            ys.append(min(float(L - a), cap))
            es.append(np.concatenate([baseline, [min(a / 250.0, 1.0)], ev]))
            us.append(int(u)); ag.append(a)
    return (np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32),
            np.stack(es).astype(np.float32), us, np.asarray(ag, dtype=float))


def cross_channel_consistency(values: np.ndarray) -> float:
    """1 - dispersion of per-channel normalised trend slopes, clipped to [0,1]."""
    t = np.arange(values.shape[0], dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-9)
    slopes = (values * t[:, None]).sum(0) / values.shape[0]
    mag = np.abs(slopes)
    if mag.mean() < 1e-6:
        return 0.0
    disp = mag.std() / (mag.mean() + 1e-9)
    return float(np.clip(1.0 - disp, 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD002")
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cols = list(SELECTED_SENSOR_COLUMNS)

    rows = []
    for fold in FOLDS:
        mpath = f"{ROOT}/experiments/ecs/{a.subset}/f{fold}_s{a.seed}"
        if not os.path.exists(f"{mpath}/like.pt"):
            print(f"missing {mpath}, skip"); continue
        ck = torch.load(f"{mpath}/like.pt", map_location="cpu")
        model = BinLikelihood()
        model.load_state_dict(ck["model"])
        meta = json.load(open(f"{mpath}/metrics.json"))
        prior = AgeConditionalPrior.__new__(AgeConditionalPrior)
        prior.cap, prior.n_bins = 125.0, N_BINS
        prior.edges = np.linspace(0.0, 125.0, N_BINS + 1)
        prior.k, prior.lam = meta["prior"]["k"], meta["prior"]["lam"]

        _, tr, va, _ = e21.split_frames(a.subset, fold)
        va_s = e21.scale_frame(a.subset, e21.fit_scaler(a.subset, tr), va)
        xva, yva, eva, units, ages = build_at_ages(va, va_s)
        lp = log_prob_np(model, xva, eva, device)

        for i, u in enumerate(units):
            g = va[va.unit == u]
            a_i = float(ages[i])
            end = int(min(a_i, len(g)))
            win = g.iloc[max(0, end - WINDOW):end][cols].to_numpy(float)
            rho = cross_channel_consistency(win)
            kappa = float(min(1.0, a_i / WINDOW))
            b = int(rul_to_bin([yva[i]])[0])
            p0 = prior.prob(a_i)
            l0 = float(np.log(p0[b]))
            l1 = float(lp[i, b])
            rows.append(dict(fold=fold, unit=int(u), age=a_i,
                             y=float(yva[i]), kappa=kappa, rho=rho,
                             l0=l0, l1=l1, delta=l1 - l0))

    d = np.array([[r["delta"], r["kappa"], r["rho"]] for r in rows])
    delta, kappa, rho = d[:, 0], d[:, 1], d[:, 2]
    if np.std(kappa) < 1e-9 or np.std(rho) < 1e-9:
        raise RuntimeError(f"degenerate regressors: std(kappa)={np.std(kappa):.2e}, "
                           f"std(rho)={np.std(rho):.2e}")
    X = np.column_stack([np.ones_like(kappa), kappa, rho])
    beta, *_ = np.linalg.lstsq(X, delta, rcond=None)
    resid = delta - X @ beta
    dof = len(delta) - 3
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_stat = beta / se
    from math import erf, sqrt
    def p_one_sided(t):   # H1: coefficient > 0
        return 1 - 0.5 * (1 + erf(t / sqrt(2)))

    out = dict(subset=a.subset, seed=a.seed, n_engines=len(rows),
               mean_delta=float(delta.mean()),
               beta=dict(intercept=float(beta[0]), kappa=float(beta[1]),
                         rho=float(beta[2])),
               se=dict(kappa=float(se[1]), rho=float(se[2])),
               t=dict(kappa=float(t_stat[1]), rho=float(t_stat[2])),
               p_one_sided=dict(kappa=float(p_one_sided(t_stat[1])),
                                rho=float(p_one_sided(t_stat[2]))),
               corr_delta_kappa=float(np.corrcoef(delta, kappa)[0, 1]),
               corr_delta_rho=float(np.corrcoef(delta, rho)[0, 1]),
               K3_pass=bool(t_stat[1] > 0 and t_stat[2] > 0),
               per_engine=rows)
    os.makedirs(f"{ROOT}/results/ecs", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/ecs/stageB_{a.subset}_s{a.seed}.json", "w"),
              indent=2)
    print(f"{a.subset} n={len(rows)} mean_delta={delta.mean():.3f}")
    print(f"  beta_kappa={beta[1]:.3f} (t={t_stat[1]:.2f}, p={p_one_sided(t_stat[1]):.4f})")
    print(f"  beta_rho  ={beta[2]:.3f} (t={t_stat[2]:.2f}, p={p_one_sided(t_stat[2]):.4f})")
    print(f"  corr(delta,kappa)={out['corr_delta_kappa']:.3f} corr(delta,rho)={out['corr_delta_rho']:.3f}")
    print(f"  H3 (both slopes > 0): {'PASS' if out['K3_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
