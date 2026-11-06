"""Learned within-condition local trends attached to the existing PE encoder.

The trend branch is invariant to a constant additive offset for each
condition/sensor, with fixed condition labels and observation masks. The
whole predictor is NOT invariant because its PE branch still sees levels.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .vendor.pe_model import CandidateModel


ARMS = ("pe50", "pe100", "w100_late", "fixed_trend", "learned_trend", "unmatched_trend")


class ConditionTrend(nn.Module):
    """Differentiable weighted fixed-effects regression at three horizons.

Regress sensor values on time after removing a separate weighted intercept
for each operating condition. Learned weights depend only on condition-
centered measurements, so they preserve the offset-cancellation property.
The fixed control uses the identical estimator with uniform weights.
"""

    def __init__(self, learned=True, match_conditions=True,
                 horizons=(20, 50, 100), n_conditions=6):
        super().__init__()
        if not horizons or min(horizons) < 2:
            raise ValueError("Trend horizons must contain at least two cycles")
        self.learned = bool(learned)
        self.match_conditions = bool(match_conditions)
        self.n_conditions = int(n_conditions)
        self.register_buffer("horizons", torch.tensor(horizons, dtype=torch.float32))
        self.weight_net = None
        if learned:
            self.weight_net = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
            nn.init.zeros_(self.weight_net[-1].weight)
            nn.init.zeros_(self.weight_net[-1].bias)

    def forward(self, history, cycles, conditions, observed, as_of):
        # history B,L,N; cycles/conditions B,L; observed B,L,N; as_of B.
        if history.ndim != 3 or observed.shape != history.shape:
            raise ValueError("history/observed must have shape (batch, time, sensors)")
        if cycles.shape != history.shape[:2] or conditions.shape != cycles.shape:
            raise ValueError("cycles/conditions must align with history")
        x = torch.where(torch.isfinite(history), history, torch.zeros_like(history))
        h = self.horizons.to(history).view(1, -1, 1, 1)
        lag = as_of.reshape(-1, 1, 1, 1).to(history) - cycles[:, None, :, None].to(history)
        valid = (observed[:, None].bool() & torch.isfinite(history[:, None])
                 & (lag >= 0) & (lag < h) & (conditions[:, None, :, None] >= 0))
        labels = conditions.clamp(min=0).long()
        if self.match_conditions:
            group = F.one_hot(labels, num_classes=self.n_conditions).to(history)
        else:
            group = torch.ones((*labels.shape, 1), dtype=history.dtype, device=history.device)
        v = valid.to(history)
        # Time is in units of the horizon; resulting slope is change/horizon.
        time = (-lag / h).expand_as(v)
        count = torch.einsum("blc,bhln->bhcn", group, v)
        safe_count = count.clamp_min(1.0)
        mean_x = torch.einsum("blc,bhln,bln->bhcn", group, v, x) / safe_count
        mean_t = torch.einsum("blc,bhln,bhln->bhcn", group, v, time) / safe_count
        center_x = x[:, None] - torch.einsum("blc,bhcn->bhln", group, mean_x)
        center_t = time - torch.einsum("blc,bhcn->bhln", group, mean_t)
        group_var = torch.einsum("blc,bhln->bhcn", group, v * center_x.square()) / safe_count
        group_sd = torch.einsum("blc,bhcn->bhln", group, (group_var + 1e-8).sqrt())
        support = torch.einsum("blc,bhcn->bhln", group, count) / h
        if self.learned:
            features = torch.stack((center_x, center_t, group_sd, support), dim=-1)
            # Bounded logits avoid fitting an effectively one-point regression.
            logits = 2.0 * torch.tanh(self.weight_net(features).squeeze(-1))
        else:
            logits = torch.zeros_like(center_x)
        weights = torch.softmax(logits.masked_fill(~valid, -1e9), dim=2) * v
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-12)
        mass = torch.einsum("blc,bhln->bhcn", group, weights).clamp_min(1e-12)
        wx = torch.einsum("blc,bhln,bln->bhcn", group, weights, x) / mass
        wt = torch.einsum("blc,bhln,bhln->bhcn", group, weights, time) / mass
        dx = x[:, None] - torch.einsum("blc,bhcn->bhln", group, wx)
        dt = time - torch.einsum("blc,bhcn->bhln", group, wt)
        leverage = (weights * dt.square()).sum(dim=2)
        identifiable = leverage > 1e-6
        slope = (weights * dt * dx).sum(dim=2) / leverage.clamp_min(1e-6)
        slope = torch.where(identifiable, slope, torch.zeros_like(slope))
        residual = dx - slope[:, :, None] * dt
        noise = ((weights * residual.square()).sum(dim=2) + 1e-8).sqrt()
        noise = torch.where(identifiable, noise, torch.zeros_like(noise))
        effective_n = 1.0 / weights.square().sum(dim=2).clamp_min(1e-12)
        effective_n = torch.where(valid.any(dim=2), effective_n, torch.zeros_like(effective_n))
        stats = torch.stack((slope, torch.log1p(noise), leverage.clamp_min(1e-12).sqrt(),
                             effective_n / self.horizons.to(history)[None, :, None]), dim=-1)
        return {"stats": stats, "slope": slope, "noise": noise,
                "leverage": leverage, "identifiable": identifiable,
                "weights": weights}


class ContrastPE(nn.Module):
    """One PE predictor with optional history-trend context (no model router)."""

    def __init__(self, arm="learned_trend", use_prior=True, num_sensors=14,
                 d_model=64, dropout=0.15):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm}; choose {ARMS}")
        self.arm = arm
        self.window = 100 if arm in ("pe100", "w100_late") else 50
        self.body = CandidateModel(num_sensors=num_sensors, window_size=self.window,
                                   d_model=d_model, num_layers=1, dropout=dropout,
                                   use_baseline=True, use_ema_features=True,
                                   use_cycle=True, use_pe=True, use_graph=False,
                                   use_dual_head=False)
        self.prior_proj = nn.Linear(2, d_model) if use_prior else None
        if self.prior_proj is not None:
            nn.init.zeros_(self.prior_proj.weight)
            nn.init.zeros_(self.prior_proj.bias)
        self.trend = None
        if arm.endswith("trend"):
            self.trend = ConditionTrend(learned=arm != "fixed_trend",
                                        match_conditions=arm != "unmatched_trend")
            self.trend_proj = nn.Sequential(nn.Linear(3 * num_sensors * 4, d_model),
                                            nn.GELU(), nn.LayerNorm(d_model),
                                            nn.Linear(d_model, d_model))
            nn.init.zeros_(self.trend_proj[-1].weight)
            nn.init.zeros_(self.trend_proj[-1].bias)

    def forward(self, history, cycles, conditions, observed, as_of, extras, prior):
        context = self.body.encode(history[:, -self.window:], extras=extras).mean(dim=1)
        if self.prior_proj is not None:
            context = context + self.prior_proj(prior)
        if self.trend is not None:
            info = self.trend(history, cycles, conditions, observed, as_of)
            context = context + self.trend_proj(info["stats"].flatten(1))
        return self.body.reg_head(context).reshape(-1)


def tangent_nasa(error):
    """NASA loss with C1 tangent continuation after an exponent of six.

    The evaluation metric always uses the exact exponential. This bounded-
    gradient training surrogate is an engineering control, not a novelty
    claim. Unlike clipping predictions, it never changes reported outputs.
    """
    scale = torch.where(error > 0, 10.0, 13.0)
    magnitude = error.abs() / scale
    q = magnitude.new_tensor(6.0)
    return torch.expm1(magnitude.clamp_max(6.0)) + torch.exp(q) * F.relu(magnitude - q)


def training_loss(prediction, target, nasa_weight=0.0):
    """Existing frozen objective, optionally plus a NASA tail surrogate.

The extra loss is an engineering control, not a claimed new method. Its
default weight is zero; run the same nonzero weight on the PE control.
No prediction clipping is used in training or primary evaluation.
"""
    error = prediction - target
    base = (error.square() * (1.0 + 0.5 * torch.where(error > 0, 1.8, 1.0))).mean()
    return base + float(nasa_weight) * tangent_nasa(error).mean()
