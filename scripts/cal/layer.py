"""Predictive distributions over the capped RUL label.

Every method maps a backbone point prediction yhat to a pmf over bins 0..125, where bin
125 is the atom P(Y = 125) that the cap convention creates.

**The bug this module exists to fix.** Evaluating a density on the integer grid and
normalising with a softmax is truncation, not censoring: the top bin ends up holding the
density at 125 instead of the surviving mass beyond it. Measured on the frozen v1
calibrator the two differ by ~8x (0.02-0.04 vs 0.19-0.28). Every capped label is then
penalised by that factor, which drags high-RUL predictions down -- the -10 cycle bias in
the [100,125] band that was previously attributed to the label convention itself.

`mode="censored"` computes bin masses as CDF differences with the last bin set to the
survival function. That is the Tobit likelihood, applied to the cap.

The shape is Gaussian rather than Student-t because PyTorch has no differentiable
Student-t CDF, and training with one family while scoring with another would be exactly
the kind of silent mismatch this module exists to remove. The cap-as-atom argument is
independent of the tail family; a heavy-tailed variant only changes which CDF is called.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtr as _ndtr_np

MAX_RUL = 125
K = MAX_RUL + 1  # bins 0..125, last one is the atom


# --------------------------------------------------------------------------- pmf
def pmf_np(mu, sigma, mode="censored"):
    """(n, K) bin masses.

    censored: p[0]=F(1), p[b]=F(b+1)-F(b) for 1..124, p[125]=1-F(125)
    density : softmax of the density on the integer grid (the v1 behaviour)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if mode == "density":
        z = (np.arange(K)[None, :] - mu[:, None]) / sigma[:, None]
        lp = -0.5 * z ** 2
        lp -= lp.max(1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(1, keepdims=True)
    Fm = _ndtr_np((np.arange(1, K)[None, :] - mu[:, None]) / sigma[:, None])
    p = np.zeros((len(mu), K), dtype=np.float64)
    p[:, 0] = Fm[:, 0]
    p[:, 1:K - 1] = np.diff(Fm, axis=1)
    p[:, K - 1] = 1.0 - Fm[:, -1]
    p = np.clip(p, 0.0, None)
    return p / p.sum(1, keepdims=True)


def atom_mass(mu, sigma):
    """The quantity the cap is supposed to carry."""
    return 1.0 - _ndtr_np((MAX_RUL - np.asarray(mu, float)) / np.asarray(sigma, float))


# --------------------------------------------------------------------------- layer
class LocationScale(nn.Module):
    """mu = yhat + 20*f(x);  sigma = softplus(g(x))*20 + 1, x = yhat/125."""

    def __init__(self, kind="mlp", hidden=16):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        elif kind == "linear":
            self.net = nn.Linear(1, 2)
        else:
            raise ValueError(kind)

    def params(self, yhat):
        out = self.net((yhat / MAX_RUL).unsqueeze(-1))
        mu = yhat + 20.0 * out[..., 0]
        sigma = F.softplus(out[..., 1].clamp(-4.0, 3.0)) * 20.0 + 1.0
        return mu, sigma


def loglik(net, yhat, y, mode="censored"):
    mu, sigma = net.params(yhat)
    if mode == "density":
        zz = (torch.arange(K, dtype=yhat.dtype, device=yhat.device).unsqueeze(0)
              - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
        lp = torch.log_softmax(-0.5 * zz ** 2, dim=-1)
        return lp.gather(-1, y.clamp(0, K - 1).long().unsqueeze(-1)).squeeze(-1)
    lo = torch.clamp(y, 0.0, MAX_RUL - 1.0)
    hi = torch.clamp(y + 1.0, 0.0, MAX_RUL)
    F_hi = torch.special.ndtr((hi - mu) / sigma)
    F_lo = torch.special.ndtr((lo - mu) / sigma)
    p = torch.where(y >= MAX_RUL, 1.0 - F_lo, F_hi - F_lo)
    return torch.log(p.clamp_min(1e-12))


def fit_layer(dev, kind="mlp", mode="censored", epochs=400, lr=0.05, ridge=1e-3, seed=0):
    torch.manual_seed(seed)
    net = LocationScale(kind).double()
    yhat = torch.tensor(np.asarray(dev["yhat"], dtype=np.float64))
    y = torch.tensor(np.asarray(dev["y"], dtype=np.float64))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = -loglik(net, yhat, y, mode).mean()
        loss = loss + ridge * sum(p.pow(2).sum() for p in net.parameters())
        loss.backward()
        opt.step()
    return net


def gen_pmf(net, yhat, mode="censored", scale=1.0):
    with torch.no_grad():
        mu, sigma = net.params(torch.tensor(np.asarray(yhat, dtype=np.float64)))
    return pmf_np(mu.numpy(), sigma.numpy() * scale, mode)


# --------------------------------------------------------------------------- intervals & scale
def interval(p, level=0.90):
    cdf = np.cumsum(p, axis=1)
    a = (1 - level) / 2
    lo = np.array([np.searchsorted(cdf[i], a) for i in range(len(p))])
    hi = np.minimum(np.array([np.searchsorted(cdf[i], 1 - a) for i in range(len(p))]), K - 1)
    return lo, hi


def cover(p, y, level=0.90):
    lo, hi = interval(p, level)
    return float(np.mean((lo <= y) & (y <= hi)))


def choose_scale(fit_fn, dev, target=0.90, n_splits=5, seed=0, grid=None):
    """Smallest sigma multiplier whose cross-fitted dev coverage reaches the target.

    Coverage is not monotone in the scale near the cap: a very wide pmf approaches the
    uniform distribution over the grid, whose central 90% excludes the atom, so the global
    argmin of the coverage gap lands on a pathological branch. Smallest-reaching is both
    sharper and stable.
    """
    grid = np.linspace(0.7, 2.5, 37) if grid is None else grid
    rng = np.random.default_rng(seed)
    n = len(dev["y"])
    folds = np.array_split(rng.permutation(n), n_splits)
    nets = []
    for f in folds:
        mask = np.ones(n, bool)
        mask[f] = False
        nets.append((fit_fn({k: dev[k][mask] for k in ("yhat", "y")}), f))
    first = None
    for s in grid:
        ps = np.zeros((n, K), dtype=np.float64)
        for net, f in nets:
            ps[f] = gen_pmf(net, dev["yhat"][f], scale=s)
        c = cover(ps, dev["y"], target)
        if first is None:
            first = (float(s), float(c))
        if c >= target:
            return float(s), float(c)
    return first
