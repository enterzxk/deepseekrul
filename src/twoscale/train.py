"""CPU training of the two scale heads on cached frozen features.

Design decision (docs/CANDIDATE_E_SPEC.md §7): the backbone and the quantile heads
stay FROZEN; only sigma_win and sigma_eng are fitted.  That isolates the mechanism,
makes the ablation trivial (sigma_eng == 1), and costs no GPU.

Objective (per mini-batch of windows):
    L = pinball(lo, hi) + lam_nll * NLL(y; ctr, half * scale) + lam_reg * (log scale)^2
where the base interval [lo0, hi0] and the centre ctr come from the frozen model,
and scale = sigma_win * sigma_eng with the identifiability constraints built into
src/twoscale/model.py.

Smoke test (synthetic features with a KNOWN engine-level difficulty):
    python -m src.twoscale.train
It must recover rho(sigma_eng, true engine error) >= 0.6 and beat the
use_engine=False ablation on Winkler.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _segment_mean(v: torch.Tensor, ids: torch.Tensor, n: int) -> torch.Tensor:
    s = torch.zeros(n, dtype=v.dtype)
    c = torch.zeros(n, dtype=v.dtype)
    s.index_add_(0, ids, v)
    c.index_add_(0, ids, torch.ones_like(v))
    return (s / c.clamp_min(1.0))[ids]


def pinball(pred, target, q):
    d = target - pred
    return torch.mean(torch.maximum(q * d, (q - 1.0) * d))


def nll(y, mu, sigma):
    return torch.mean(torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2)


class Heads(nn.Module):
    def __init__(self, d: int, hidden: int = 64, use_engine: bool = True):
        super().__init__()
        self.use_engine = use_engine
        self.win = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.eng = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, h, ids, n_eng):
        win = torch.nn.functional.softplus(self.win(h)).reshape(-1) + 1e-3
        win = win / _segment_mean(win, ids, n_eng)
        if not self.use_engine:
            return win, torch.ones_like(win)
        pooled = torch.zeros(n_eng, h.shape[-1])
        cnt = torch.zeros(n_eng)
        pooled.index_add_(0, ids, h.detach())
        cnt.index_add_(0, ids, torch.ones(h.shape[0]))
        pooled = pooled / cnt.clamp_min(1.0).unsqueeze(-1)
        eng = torch.nn.functional.softplus(self.eng(pooled)).reshape(-1) + 1e-3
        eng = eng / eng.mean()
        return win, eng[ids]


def fit_two_scale(h, ids, ctr, half, y, n_eng, *, use_engine=True, epochs=200,
                  lr=3e-3, lam_nll=0.1, seed=0):
    torch.manual_seed(seed)
    h_t = torch.as_tensor(h, dtype=torch.float32)
    model = Heads(h_t.shape[-1], use_engine=use_engine)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ctr_t = torch.as_tensor(ctr, dtype=torch.float32)
    half_t = torch.as_tensor(half, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    ids_t = torch.as_tensor(ids, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        win, eng = model(h_t, ids_t, n_eng)
        scale = win * eng
        lo = ctr_t - half_t * scale
        hi = ctr_t + half_t * scale
        loss = pinball(lo, y_t, 0.05) + pinball(hi, y_t, 0.95) + lam_nll * nll(y_t, ctr_t, half_t * scale)
        loss.backward()
        opt.step()
    with torch.no_grad():
        win, eng = model(h_t, ids_t, n_eng)
        return model, (win * eng).numpy(), eng.numpy()


def _smoke():
    rng = np.random.default_rng(0)
    n_eng, per, d = 60, 40, 16
    ids = np.repeat(np.arange(n_eng), per)
    # engine-level difficulty is a linear function of the pooled feature (learnable)
    diff = rng.normal(0, 0.35, n_eng)
    h = rng.normal(0, 1, (n_eng * per, d)).astype(np.float32)
    h[:, 0] += np.repeat(diff, per) * 3.0
    ctr = np.zeros(n_eng * per, dtype=np.float32)
    half = np.ones(n_eng * per, dtype=np.float32)
    y = (rng.normal(0, 1, n_eng * per) * np.repeat(np.exp(diff), per)).astype(np.float32)

    res = {}
    for use_eng in (True, False):
        model, scale, eng = fit_two_scale(h, ids, ctr, half, y, n_eng, use_engine=use_eng)
        rmse_eng = np.array([np.sqrt(np.mean((y[ids == e]) ** 2)) for e in range(n_eng)])
        if use_eng:
            pred = np.array([eng[ids == e][0] for e in range(n_eng)])
            rho = float(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(rmse_eng)))[0, 1])
        else:
            rho = float("nan")
        w = float(np.mean((2 * half * scale) + 20 * (np.abs(y - ctr) - half * scale).clip(min=0)))
        res[use_eng] = (rho, w)
        print(f"use_engine={use_eng}: rho(sigma_eng, engine RMSE)={rho:+.3f}  Winkler={w:.3f}")

    ok = (res[True][0] >= 0.60) and (res[True][1] <= res[False][1])
    print("\nRESULT:", "PASS" if ok else "FAIL",
          "(engine head must recover rho>=0.6 and beat the ablation on Winkler)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_smoke())
