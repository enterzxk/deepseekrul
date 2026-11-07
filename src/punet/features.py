"""Align the causal lifetime prior with dataset windows.

The prior is read through StaircasePrior.lookup, which consumes the latest
cutoff c <= t for an engine and never interpolates towards the future. This
module only wraps that lookup into batched tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class PriorFeatures:
    """Batched prior features for one batch of windows.

    Attributes:
        mu: prior mean lifetime mu_T, shape (B,).
        sigma: prior standard deviation sigma_T, shape (B,).
        tau: degradation clock t / mu_T, shape (B,).
        valid: False where the engine had no usable cutoff (median fallback).
    """

    mu: torch.Tensor
    sigma: torch.Tensor
    tau: torch.Tensor
    valid: torch.Tensor

    def to(self, device) -> "PriorFeatures":
        return PriorFeatures(
            mu=self.mu.to(device),
            sigma=self.sigma.to(device),
            tau=self.tau.to(device),
            valid=self.valid.to(device),
        )

    def __len__(self) -> int:
        return int(self.mu.shape[0])


class PriorFeatureBuilder:
    """Turn (unit, cutoff) pairs into PriorFeatures.

    Args:
        prior: a fitted StaircasePrior, or any object exposing
            lookup(outer_fold, unit, t) -> (mu, sigma, flag, cutoff).
        outer_fold: outer cross-fitting fold index used by the prior table.
        sigma_floor: lower bound applied to sigma to keep the scale finite.
    """

    def __init__(self, prior, outer_fold: int = 0, sigma_floor: float = 1e-3) -> None:
        self.prior = prior
        self.outer_fold = int(outer_fold)
        self.sigma_floor = float(sigma_floor)

    def build(self, units: Sequence[int], cutoffs: Sequence[int]) -> PriorFeatures:
        if len(units) != len(cutoffs):
            raise ValueError("units and cutoffs must have the same length")

        mu_list = []
        sigma_list = []
        tau_list = []
        valid_list = []

        for unit, t in zip(units, cutoffs):
            t = int(t)
            mu, sigma, flag, used_cutoff = self.prior.lookup(self.outer_fold, int(unit), t)
            if used_cutoff > t:  # defensive: the prior already asserts this (T4)
                raise ValueError(f"prior consumed cutoff {used_cutoff} after window end {t}")
            sigma = max(float(sigma), self.sigma_floor)
            mu_safe = float(mu) if float(mu) > 0 else 1.0
            mu_list.append(float(mu))
            sigma_list.append(sigma)
            tau_list.append(t / mu_safe)
            valid_list.append(flag == 0)

        return PriorFeatures(
            mu=torch.tensor(mu_list, dtype=torch.float32),
            sigma=torch.tensor(sigma_list, dtype=torch.float32),
            tau=torch.tensor(tau_list, dtype=torch.float32),
            valid=torch.tensor(valid_list, dtype=torch.bool),
        )
