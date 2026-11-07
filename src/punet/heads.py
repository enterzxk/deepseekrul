"""Scale and bias heads that consume prior uncertainty."""

from __future__ import annotations

import math

import torch
from torch import nn


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class ScaleHead(nn.Module):
    """Predict a strictly positive per-sample interval scale.

    The scale is w = (sigma + eps) ** a * exp(MLP([h, tau])) with
    a = softplus(a_raw) > 0, so d w / d sigma > 0 holds by construction.

    Initialisation makes a close to 1 and exp(MLP) == 1, i.e. training starts
    from a pure power law, which is exactly ablation A5's form.
    """

    def __init__(self, d_model: int, hidden: int = 64, eps: float = 1e-3,
                 log_scale_min: float = -6.9, log_scale_max: float = 6.9,
                 simple: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)
        self.simple = bool(simple)
        self.log_scale_min = float(log_scale_min)   # w >= ~1e-3
        self.log_scale_max = float(log_scale_max)   # w <= ~1e3
        self.a_raw = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.mlp = None
        if not self.simple:
            self.mlp = nn.Sequential(
                nn.Linear(d_model + 1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    @property
    def exponent(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.a_raw)

    def forward(self, h: torch.Tensor, sigma: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Args: h (B, D), sigma (B,), tau (B,). Returns w (B,)."""
        if h.ndim != 2:
            raise ValueError(f"h must be (B, D), got {tuple(h.shape)}")
        log_sigma = torch.log(sigma.clamp_min(self.eps) + self.eps)
        if self.mlp is None:
            log_mod = torch.zeros_like(log_sigma)
        else:
            log_mod = self.mlp(torch.cat([h, tau.unsqueeze(-1)], dim=-1)).squeeze(-1)
        log_w = self.exponent * log_sigma + log_mod
        return torch.exp(log_w.clamp(self.log_scale_min, self.log_scale_max))


class BiasHead(nn.Module):
    """Zero-initialised point-prediction correction.

    The final layer is zeroed, so b == 0 at step 0 and the model is bit-wise
    equivalent to the frozen Gate 0 baseline.
    """

    def __init__(self, d_model: int, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Args: h (B, D), sigma (B,). Returns b (B,)."""
        return self.mlp(torch.cat([h, sigma.unsqueeze(-1)], dim=-1)).squeeze(-1)
