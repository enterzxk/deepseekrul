"""EAC: Engine-Adaptive Conditioning for the frozen PE candidate (V0).

The network reads the unit's own first k cycles E (k x N), computes per-sensor
statistics (mean, std, first-last difference), embeds them into a unit vector g,
and emits per-sensor FiLM parameters (gamma_s, beta_s) applied right after the
shared sensor mapping and before the per-sensor temporal convolution:

    h_s <- gamma_s * h_s + beta_s

Leakage constraints (spec section 2):
  * E only ever contains rows with cycle index < k of that unit -- the builder
    in scripts/run_eac.py enforces this; this module never sees RUL or total life.
  * The last FiLM layer is zero-initialised, so at init gamma=1, beta=0 and the
    wrapped model is bit-identical to V0 (unit test in tests/test_eac.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EACGenerator(nn.Module):
    """Per-sensor FiLM parameters from the unit's early history E (B, k, N)."""

    def __init__(self, num_sensors: int = 14, d_model: int = 64, n_stats: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.stat_proj = nn.Linear(n_stats, d_model)
        self.unit_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.sensor_embed = nn.Embedding(num_sensors, d_model)
        self.film_head = nn.Linear(2 * d_model, 2 * d_model)
        nn.init.zeros_(self.film_head.weight)
        nn.init.zeros_(self.film_head.bias)

    def forward(self, e: torch.Tensor):
        """e: (B, k', N) with k' <= k (training randomises k' in [20, k])."""
        mu = e.mean(dim=1)                                  # (B, N)
        sd = e.std(dim=1)                                   # (B, N)
        fl = e[:, -1, :] - e[:, 0, :]                       # (B, N)
        stats = torch.stack([mu, sd, fl], dim=-1)           # (B, N, 3)
        se = F.gelu(self.stat_proj(stats))                  # (B, N, D)
        g = self.unit_mlp(se.mean(dim=1))                   # (B, D)
        b_size, n_sens = mu.shape
        ids = torch.arange(n_sens, device=e.device)
        sid = self.sensor_embed(ids).unsqueeze(0).expand(b_size, -1, -1)   # (B, N, D)
        inp = torch.cat([g.unsqueeze(1).expand(-1, n_sens, -1), sid], dim=-1)
        gb = self.film_head(inp)                            # (B, N, 2D)
        gamma, beta = gb.chunk(2, dim=-1)
        return 1.0 + gamma, beta                            # identity at init


class SensorInFiLM(nn.Module):
    """Wraps the shared sensor mapping Linear and applies FiLM from E.

    The wrapped model must set ``wrapper.current_E`` (B, k', N) before every
    forward; ``None`` disables conditioning (used by the identity test and by
    V0 runs that reuse the same class).
    """

    def __init__(self, inner: nn.Module, num_sensors: int, d_model: int):
        super().__init__()
        self.inner = inner
        self.gen = EACGenerator(num_sensors, d_model)
        self.current_E: torch.Tensor | None = None

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        h = self.inner(stacked)                             # (B, L, N, D)
        if self.current_E is None:
            return h
        gamma, beta = self.gen(self.current_E)              # (B, N, D)
        return h * gamma.unsqueeze(1) + beta.unsqueeze(1)


def install_eac(model: nn.Module, num_sensors: int = 14, d_model: int = 64) -> SensorInFiLM:
    """Wrap ``model.sensor_in`` in place.  Call BEFORE creating the optimiser so
    the FiLM parameters are registered and trained.  Installing after the body
    is constructed also guarantees the body weights match a same-seed V0."""
    wrapper = SensorInFiLM(model.sensor_in, num_sensors, d_model)
    model.sensor_in = wrapper
    return wrapper


def derangement(n: int, rng: np.random.Generator, tries: int = 100) -> np.ndarray:
    """A permutation of range(n) with no fixed points (for the V6 control)."""
    for _ in range(tries):
        p = rng.permutation(n)
        if n == 1:
            raise ValueError("V6 needs at least 2 units")
        if not (p == np.arange(n)).any():
            return p
    raise RuntimeError("no derangement found")
