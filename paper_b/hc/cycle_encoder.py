"""Cycle encoders for the Stage C study (task book §6, variants A0-A5).

Every encoder maps a *cycle scalar in cycles* to a d_model vector that is added
to the pooled spatio-temporal representation. The scalar is produced by
``data_hc.cycle_scalar`` so the clipping / normalisation question stays in one
place.

Implementation note that matters for the A3 sub-comparison
---------------------------------------------------------
A pure Fourier encoder is invariant to rescaling the scalar: angles are
2*pi*c/period, so feeding c/500 with periods p/500 gives exactly the same
angles. "Clipped vs unclipped" therefore only differs once an engine passes the
clip point, and "normalised but not saturating" cannot be expressed by
rescaling alone. A3c is consequently implemented as the same Fourier family
with the period bank widened (25/50/125/250 -> 50/100/250/500), which is the
change that actually removes the within-lifetime phase wrap-around. This is
recorded in METHOD_SPECIFICATION.md rather than hidden.
"""
import math

import torch
import torch.nn as nn

# Frozen period bank of the original DSPT.
PERIODS_FROZEN = (25.0, 50.0, 125.0, 250.0)
# Wider bank used by A3c: the longest period exceeds any observed engine life.
PERIODS_WIDE = (50.0, 100.0, 250.0, 500.0)
# Fixed knots for the monotone age spline (cycles). Protocol constants, not
# fitted from data, so no label information can leak into the encoding.
SPLINE_KNOTS = (0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0)
SPLINE_SCALE = 100.0


class FourierCycle(nn.Module):
    """Original DSPT Fourier encoding, applied to a cycle scalar."""

    def __init__(self, d_model, periods=PERIODS_FROZEN):
        super().__init__()
        self.register_buffer('periods', torch.tensor(periods, dtype=torch.float32))
        self.proj = nn.Linear(2 * len(periods), d_model)
        self.out_dim = d_model

    def forward(self, c):
        ang = (2.0 * math.pi) * c.unsqueeze(1) / self.periods.view(1, -1)
        return self.proj(torch.cat([torch.sin(ang), torch.cos(ang)], dim=1))


class ScalarCycle(nn.Module):
    """Linear projection of a single scalar (raw cycle, or log1p(cycle))."""

    def __init__(self, d_model, log=False):
        super().__init__()
        self.log = bool(log)
        self.proj = nn.Linear(1, d_model)
        self.out_dim = d_model

    def forward(self, c):
        x = torch.log1p(torch.clamp(c, min=0.0)) if self.log else c
        return self.proj(x.unsqueeze(1))


class SplineAge(nn.Module):
    """Monotone piecewise-linear age basis, linearly projected.

    Each basis function max(0, c - knot)/scale is non-decreasing in c, so the
    encoding is monotone in age before the (unconstrained) projection. The
    projection is what the model learns; the monotonicity of the *basis* is
    what makes the age axis ordered rather than an arbitrary embedding.
    """

    def __init__(self, d_model, knots=SPLINE_KNOTS, scale=SPLINE_SCALE):
        super().__init__()
        self.register_buffer('knots', torch.tensor(knots, dtype=torch.float32))
        self.scale = float(scale)
        self.proj = nn.Linear(len(knots) + 1, d_model)
        self.out_dim = d_model

    def basis(self, c):
        h = torch.relu(c.unsqueeze(1) - self.knots.view(1, -1)) / self.scale
        return torch.cat([torch.ones_like(c).unsqueeze(1), h], dim=1)

    def forward(self, c):
        return self.proj(self.basis(c))


def build_cycle_encoder(cycle_mode, d_model, period_bank='frozen'):
    if cycle_mode == 'none':
        return None
    if cycle_mode == 'raw':
        return ScalarCycle(d_model, log=False)
    if cycle_mode == 'log':
        return ScalarCycle(d_model, log=True)
    if cycle_mode == 'fourier':
        periods = PERIODS_WIDE if period_bank == 'wide' else PERIODS_FROZEN
        return FourierCycle(d_model, periods)
    if cycle_mode == 'spline':
        return SplineAge(d_model)
    raise KeyError(f'unknown cycle_mode {cycle_mode!r}')
