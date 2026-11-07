"""Conformalized quantile regression (CQR) heads for PU-Net.

Instead of scaling one global conformal quantile by a scalar sigma, this module
lets the frozen backbone's representation predict the conditional 5% / 95%
quantiles of RUL directly, then conformalises the resulting interval with the
standard CQR score (Romano et al., 2019):

    E_i = max( q_lo(x_i) - y_i , y_i - q_hi(x_i) )
    q_hat = Quantile_{(1-delta)(1+1/n)}( {E_i} )
    C(x) = [ q_lo(x) - q_hat , q_hi(x) + q_hat ]

Monotonicity q_lo <= q_hi is structural: the head predicts a centre and a
non-negative half-width.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class QuantileHeads(nn.Module):
    """Predict a centre and a non-negative half-width from (h, sigma, tau)."""

    def __init__(self, d_model: int, hidden: int = 64, init_half_width: float = 10.0) -> None:
        super().__init__()
        self.centre_head = nn.Sequential(
            nn.Linear(d_model + 2, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.half_head = nn.Sequential(
            nn.Linear(d_model + 2, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        # start from a sensible interval instead of a degenerate point
        nn.init.zeros_(self.centre_head[-1].weight)
        nn.init.zeros_(self.centre_head[-1].bias)
        nn.init.zeros_(self.half_head[-1].weight)
        nn.init.constant_(self.half_head[-1].bias, float(np.log(np.expm1(init_half_width))))

    def forward(self, h: torch.Tensor, sigma: torch.Tensor, tau: torch.Tensor):
        z = torch.cat([h, sigma.unsqueeze(-1), tau.unsqueeze(-1)], dim=-1)
        centre = self.centre_head(z).squeeze(-1)
        half = F.softplus(self.half_head(z).squeeze(-1))
        return centre - half, centre + half


def train_quantile_heads(
    heads: QuantileHeads,
    h: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
    y: torch.Tensor,
    *,
    q_lo: float = 0.05,
    q_hi: float = 0.95,
    epochs: int = 300,
    lr: float = 1e-2,
    device: str = "cpu",
    seed: int = 0,
):
    """Fit the two quantile levels with pinball loss."""
    torch.manual_seed(seed)
    h, sigma, tau, y = h.to(device), sigma.to(device), tau.to(device), y.to(device).reshape(-1)
    heads = heads.to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=1e-4)

    def pinball(pred, target, q):
        d = target - pred
        return torch.mean(torch.maximum(q * d, (q - 1.0) * d))

    history = []
    for _ in range(epochs):
        lo, hi = heads(h, sigma, tau)
        loss = pinball(lo, y, q_lo) + pinball(hi, y, q_hi)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        lo, hi = heads(h, sigma, tau)
    return heads, lo.cpu(), hi.cpu(), history[-1]


class CQRCalibrator:
    """Split conformal on the CQR score."""

    def __init__(self, delta: float = 0.1) -> None:
        self.delta = float(delta)
        self.q_hat = None

    @staticmethod
    def scores(y, lo, hi) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        lo = np.asarray(lo, dtype=np.float64).reshape(-1)
        hi = np.asarray(hi, dtype=np.float64).reshape(-1)
        return np.maximum(lo - y, y - hi)

    def fit(self, scores) -> "CQRCalibrator":
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        n = s.size
        level = min(1.0, (1.0 - self.delta) * (1.0 + 1.0 / n))
        self.q_hat = float(np.quantile(s, level, method="higher"))
        return self

    def interval(self, lo, hi):
        lo = np.asarray(lo, dtype=np.float64).reshape(-1)
        hi = np.asarray(hi, dtype=np.float64).reshape(-1)
        return lo - self.q_hat, hi + self.q_hat
