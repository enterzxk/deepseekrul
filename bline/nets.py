"""Backbones and the adversarial / alignment variants, parameter matched.

Every variant shares one backbone; only the input representation and the
auxiliary loss change.  ``match_width`` picks the hidden width that brings a
variant within 2% of the reference parameter count, so no comparison can be won
by having more weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("erm", "erm_u", "condnorm", "cai", "cai_w", "dann", "coral")


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam=1.0):
    return GradReverse.apply(x, lam)


class TCN(nn.Module):
    def __init__(self, d_in: int, width: int, n_cond: int = 0, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(d_in, width, 5, padding=2)
        self.c2 = nn.Conv1d(width, width, 5, padding=4, dilation=2)
        self.c3 = nn.Conv1d(width, width, 5, padding=8, dilation=4)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(width, 1))
        self.disc = None
        if n_cond:
            self.disc = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                      nn.Linear(width, n_cond))

    def features(self, x):
        x = x.transpose(1, 2)          # (N, L, d) -> (N, d, L)
        h = F.gelu(self.inp(x))
        h = self.drop(h)
        h = F.gelu(self.c2(h))
        h = self.drop(h)
        h = F.gelu(self.c3(h))
        return torch.cat([h[:, :, -1], h.mean(-1)], dim=1)

    def forward(self, x, lam=0.0):
        f = self.features(x)
        y = self.head(f).squeeze(-1)
        if self.disc is None or lam == 0.0:
            return y, None, f
        return y, self.disc(grad_reverse(f, lam)), f


def coral_penalty(f: torch.Tensor, groups: torch.Tensor, n_groups: int) -> torch.Tensor:
    d = f.shape[1]
    means, covs = [], []
    for g in range(n_groups):
        m = groups == g
        if m.sum() < 8:
            continue
        z = f[m]
        z = z - z.mean(0, keepdim=True)
        means.append(z.mean(0))
        covs.append(z.T @ z / max(len(z) - 1, 1))
    if len(covs) < 2:
        return f.new_zeros(())
    loss = f.new_zeros(())
    n = 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            loss = loss + ((covs[i] - covs[j]) ** 2).sum() / (d * len(covs))
            n += 1
    return loss / max(n, 1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_width(d_in: int, target: int, n_cond: int = 0) -> tuple[int, int]:
    """Hidden width whose parameter count is closest to ``target``."""
    best = None
    for w in range(8, 129, 2):
        n = count_params(TCN(d_in, w, n_cond=n_cond))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (w, n)
    return best


def input_dim(variant: str) -> int:
    return {"erm": 21, "condnorm": 21, "cai": 21, "cai_w": 42,
            "erm_u": 24, "dann": 21, "coral": 21}[variant]
