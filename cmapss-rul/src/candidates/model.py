"""Stage-2 candidate architecture for the IEEE TIM track.

One flexible model covers Tracks A-D behind explicit switches so the
tournament can attribute every point of validation RMSE to a component:

  Track A  health-baseline deviation | causal EMA channels | cycle coordinate
  Track B  channel-independent per-sensor encoder -> sensor-wise dynamic graph
           attention (zero-init gate) -> shared temporal transformer
  Track C  knee-aware dual head (health-state classification + regression)
           and a differentiable soft-NASA surrogate
  Track D  RUL-guided contrastive regularisation on the pooled embedding

Nothing here reads the test split: baselines and cycle coordinates are derived
from the input window and its engine metadata only.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class ModelEMA:
    """Exponential moving average of model weights (Track B)."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def store_and_apply(self, model: nn.Module) -> None:
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self.backup:
            model.load_state_dict(self.backup)
            self.backup = {}


class SensorGraphAttention(nn.Module):
    """Dynamic sensor-wise graph attention with a zero-init condition gate."""

    def __init__(self, d_model: int, num_heads: int = 4, cond_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(cond_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h: torch.Tensor, cond_t: torch.Tensor) -> torch.Tensor:
        # h: (B*L, N, D); cond_t: (B*L, 1, cond_dim)
        update, _ = self.attn(h, h, h, need_weights=False)
        # Zero-initialised gate: alpha == 0 exactly at step 0, and the norm is
        # applied to the UPDATE ONLY, so the block is a strict identity at
        # initialisation (LayerNorm(h + 0) would not be).
        alpha = 0.5 * (torch.sigmoid(self.gate(cond_t)) - 0.5)
        return h + alpha * self.norm(update)


class CandidateModel(nn.Module):
    """Channel-independent, health-calibrated, knee-aware RUL regressor."""

    def __init__(
        self,
        num_sensors: int = 14,
        window_size: int = 50,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
        use_baseline: bool = True,
        use_ema_features: bool = True,
        use_cycle: bool = True,
        use_graph: bool = True,
        use_dual_head: bool = True,
        max_rul: float = 125.0,
        use_sensor_id: bool = False,
        use_independent_channels: bool = False,
        use_reliability: bool = False,
    ) -> None:
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.use_baseline = bool(use_baseline)
        self.use_ema_features = bool(use_ema_features)
        self.use_cycle = bool(use_cycle)
        self.use_graph = bool(use_graph)
        self.use_dual_head = bool(use_dual_head)
        self.max_rul = float(max_rul)
        self.use_sensor_id = bool(use_sensor_id)
        self.use_independent_channels = bool(use_independent_channels)
        self.use_reliability = bool(use_reliability)

        per_sensor = 1 + int(use_baseline) + int(use_ema_features)
        if self.use_independent_channels:
            # one projection per physical sensor (parameter matched below)
            self.sensor_in = nn.ModuleList(
                [nn.Linear(per_sensor, d_model) for _ in range(self.num_sensors)]
            )
        else:
            self.sensor_in = nn.Linear(per_sensor, d_model)
        if self.use_sensor_id:
            self.sensor_embed = nn.Embedding(self.num_sensors, d_model)
        if self.use_reliability:
            # reliability of the health reference, from conditioning + evidence count
            self.reliability = nn.Sequential(
                nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1)
            )
        self.time_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.time_norm = nn.LayerNorm(d_model)
        if use_cycle:
            self.cycle_proj = nn.Linear(1, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_graph:
            self.graph = SensorGraphAttention(d_model, num_heads, cond_dim=3, dropout=dropout)

        # Independent layers: copying the same module into a ModuleList would
        # silently share every weight between positions.
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=num_heads, dim_feedforward=2 * d_model,
                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.temporal_pe = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.normal_(self.temporal_pe, std=0.02)

        self.sensor_score = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
        if use_dual_head:
            self.cls_head = nn.Linear(d_model, 1)
        nn.init.constant_(self.reg_head[-1].bias, 2.0)

    @staticmethod
    def _ema(x: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
        # causal exponential moving average along time
        out = torch.empty_like(x)
        acc = x[:, 0]
        out[:, 0] = acc
        for t in range(1, x.shape[1]):
            acc = beta * acc + (1.0 - beta) * x[:, t]
            out[:, t] = acc
        return out

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        extras: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        # x: (B, L, N); extras: (B, N + 1) = baseline per sensor + cycle coord
        batch, length, sensors = x.shape
        if extras is not None:
            baseline = extras[:, :sensors].unsqueeze(1)  # (B, 1, N)
            cycle = extras[:, sensors : sensors + 1]  # (B, 1)
            ref_count = extras[:, sensors + 1 : sensors + 2]  # (B, 1)
        else:
            baseline = x[:, :1, :]
            cycle = x.new_zeros(batch, 1)
            ref_count = x.new_ones(batch, 1)

        feats = [x]
        if self.use_baseline:
            feats.append(x - baseline)
        if self.use_ema_features:
            feats.append(self._ema(x, 0.5))
        if cond is not None:
            cond_t = cond if cond.ndim == 3 else cond.unsqueeze(1).expand(-1, length, -1)
        else:
            cond_t = x.new_zeros(batch, length, 3)
        if self.use_reliability:
            # Reliability-aware degradation representation: mix the reference
            # residual with the raw measurement according to how trustworthy the
            # reference is (evidence count + operating condition).
            evidence = torch.cat(
                [ref_count.unsqueeze(1).expand(-1, length, -1),
                 cycle.unsqueeze(1).expand(-1, length, -1)], dim=-1
            )  # (B, L, 2): reference evidence count, cycle position
            weight = torch.sigmoid(self.reliability(evidence))  # (B, L, 1)
            residual = x - baseline
            feats = [x, weight * residual + (1.0 - weight) * x]
            if self.use_ema_features:
                feats.append(self._ema(x, 0.5))
            stacked = torch.stack(feats, dim=-1)
        else:
            stacked = torch.stack(feats, dim=-1)  # (B, L, N, F)
        if self.use_independent_channels:
            h = torch.stack(
                [self.sensor_in[i](stacked[:, :, i, :]) for i in range(sensors)], dim=2
            )
        else:
            h = self.sensor_in(stacked)  # (B, L, N, D)

        # channel-independent temporal conv: (B*N, D, L)
        hh = h.permute(0, 2, 3, 1).reshape(batch * sensors, self.d_model, length)
        hh = F.gelu(self.time_conv(hh))
        conv_out = hh.reshape(batch, sensors, self.d_model, length).permute(0, 3, 1, 2)
        h = self.time_norm(h + self.dropout(conv_out))

        if self.use_sensor_id:
            ids = torch.arange(sensors, device=x.device)
            h = h + self.sensor_embed(ids).view(1, 1, sensors, -1)
        if self.use_graph:
            flat = h.reshape(batch * length, sensors, self.d_model)
            flat_cond = cond_t.reshape(batch * length, 1, 3)
            flat = self.graph(flat, flat_cond)
            h = flat.reshape(batch, length, sensors, self.d_model)

        # shared temporal transformer across sensors
        tokens = h.permute(0, 2, 1, 3).reshape(batch * sensors, length, self.d_model)
        tokens = tokens + self.temporal_pe
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = tokens.reshape(batch, sensors, length, self.d_model).permute(0, 2, 1, 3)

        # sensor readout at the very end (after temporal modelling)
        score = torch.softmax(self.sensor_score(tokens).squeeze(-1), dim=-1)  # (B, L, N)
        pooled = (score.unsqueeze(-1) * tokens).sum(dim=2)  # (B, L, D)
        pooled = pooled + self.cycle_proj(cycle).unsqueeze(1) if self.use_cycle else pooled
        pooled = self.norm(pooled)

        context = pooled.mean(dim=1)
        prediction = self.reg_head(context)
        raw_traj = self.reg_head(pooled).squeeze(-1)

        out = {"prediction": prediction, "trajectory": raw_traj, "embedding": context}
        if self.use_dual_head:
            out["logit"] = self.cls_head(context)
        return out


def soft_nasa_loss(
    delta: torch.Tensor,
    early_scale: float = 13.0,
    late_scale: float = 10.0,
    cap_early: float = 78.0,
    cap_late: float = 60.0,
    tail_slope: float = 0.05,
) -> torch.Tensor:
    """Differentiable, sign-correct surrogate of the NASA asymmetric score.

    With d = prediction - target the true score is

        d < 0 (early): exp(-d / 13) - 1
        d >= 0 (late): exp( d / 10) - 1

    so over-estimating RUL (late, safety critical) is penalised more steeply
    than under-estimating. Both terms are exactly 0 at d = 0 and strictly
    increasing in |d|. Inside the exponent the magnitude is capped (78 / 60
    cycles) for numerical stability, and a small linear tail keeps a non-zero
    gradient beyond the cap instead of saturating to a constant.
    """
    early_mag = torch.clamp(-delta, min=0.0)
    late_mag = torch.clamp(delta, min=0.0)
    early_term = torch.expm1(torch.clamp(early_mag, max=cap_early) / early_scale)
    early_term = early_term + tail_slope * torch.relu(early_mag - cap_early)
    late_term = torch.expm1(torch.clamp(late_mag, max=cap_late) / late_scale)
    late_term = late_term + tail_slope * torch.relu(late_mag - cap_late)
    return torch.mean(torch.where(delta < 0, early_term, late_term))


class TransformerWithFeatures(nn.Module):
    """Strong Transformer with the Track A feature set (fairness control).

    Same backbone family as baselines.VanillaTransformer, but fed the three
    feature channels (raw, healthy-baseline residual, causal EMA) so that
    feature contribution and architecture contribution can be separated.
    """

    def __init__(self, num_sensors: int = 14, window_size: int = 50, d_model: int = 88,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.proj = nn.Linear(3 * num_sensors, d_model)
        self.pe = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.normal_(self.pe, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                       dim_feedforward=2 * d_model, dropout=dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(int(num_layers))
        ])
        self.score = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
        nn.init.constant_(self.head[-1].bias, 2.0)

    @staticmethod
    def _ema(x: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
        out = torch.empty_like(x)
        acc = x[:, 0]
        out[:, 0] = acc
        for t in range(1, x.shape[1]):
            acc = beta * acc + (1.0 - beta) * x[:, t]
            out[:, t] = acc
        return out

    def forward(self, x, cond=None, extras=None, return_diagnostics=False):
        if extras is not None:
            baseline = extras[:, : self.num_sensors].unsqueeze(1)
        else:
            baseline = x[:, :1, :]
        feats = torch.cat([x, x - baseline, self._ema(x, 0.5)], dim=-1)
        h = self.proj(feats) + self.pe
        for layer in self.layers:
            h = layer(h)
        weights = torch.softmax(self.score(h).squeeze(-1), dim=-1)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
        out = self.head(torch.cat([h[:, -1, :], pooled], dim=-1))
        return {"prediction": out, "trajectory": None, "embedding": pooled}
