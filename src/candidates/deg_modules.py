"""Degradation-conditioned modules for IDA-PE Module B / C (frozen spec).

Module B (Degradation-Conditioned Temporal Attention):
    attention logits A = QK^T/sqrt(d) + B_deg, where
    B_deg[b, head, i, j] = alpha[b, head] * p_head[i] * q_head[j]
    (rank-1 per-head temporal bias scaled by a zero-init linear readout of
    the tau condition vector). Initialisation is EXACTLY the stock PE
    Transformer because alpha == 0 at step 0.

Module C (Degradation-Aware Sensor Mixer):
    after the temporal transformer, before sensor pooling,
    H_new = H + g(cond) * (U V^T H), U/V (N, rank=4) static bases and
    g a zero-init linear gate on the condition vector (tanh-bounded).
    Static control: same gate weights fed a constant vector, so capacity
    matches while condition information is removed.

Both modules consume the same 9-dim condition vector as M1p
(phi(tau) ++ u), with arm-level variants: individualised tau, absolute
cycle, shuffled tau, static (constant input).

This file deliberately does NOT touch src/candidates/model.py (in use by
running queues); DegCandidateModel subclasses CandidateModel and overrides
encode() with a verbatim copy carrying two marked insertion points.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from candidates.model import CandidateModel  # noqa: F401  (re-exported)


class DegAttnEncoderLayer(nn.Module):
    """norm-first TransformerEncoderLayer (gelu, batch_first) whose
    self-attention accepts an additive per-window bias of shape
    (B, L, L), broadcast over heads via repeat_interleave."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.nhead = int(nhead)

    def forward(self, x: torch.Tensor,
                deg_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if deg_bias is None:
            mask = None
        else:
            mask = deg_bias.repeat_interleave(self.nhead, dim=0)  # (B*h, L, L)
        y = self.norm1(x)
        attn, _ = self.self_attn(y, y, y, attn_mask=mask, need_weights=False)
        x = x + self.dropout1(attn)
        y = self.norm2(x)
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(y)))))
        return x


class DegAttentionBias(nn.Module):
    """Module B condition -> (B, L, L) attention bias.

    B_deg[b,i,j] = sum_h alpha[b,h] * p[h,i] * q[h,j]. p/q are per-head
    temporal bases (rank-1 per head), alpha is a zero-init linear readout
    of the condition vector, so B_deg == 0 exactly at initialisation.
    Parameters: 2*h*L + (cond_dim*h + h); with h=4, L=50, cond=9 -> 440.
    """

    def __init__(self, cond_dim: int = 9, n_heads: int = 4, window: int = 50) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.randn(n_heads, window) * 0.02)
        self.q = nn.Parameter(torch.randn(n_heads, window) * 0.02)
        self.alpha = nn.Linear(cond_dim, n_heads)
        nn.init.zeros_(self.alpha.weight)
        nn.init.zeros_(self.alpha.bias)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim) -> bias (B, L, L)
        a = self.alpha(cond)                                     # (B, h)
        base = torch.einsum("hi,hj->hij", self.p, self.q)        # (h, L, L)
        return torch.einsum("bh,hij->bij", a, base)              # (B, L, L)


class DegSensorMixer(nn.Module):
    """Module C: low-rank inter-sensor interaction with a condition gate.

    H_new[b,l,n,:] = H[b,l,n,:] + g[b,n] * sum_m (U V^T)[n,m] H[b,l,m,:],
    g = tanh(Gate(cond)) with Gate zero-init (identity at step 0).
    The 'static' arm feeds the same gate a constant vector (capacity
    matched, condition-free). Parameters (rank=4): U,V 112 + gate 140.
    """

    def __init__(self, n_sensors: int = 14, rank: int = 4,
                 cond_dim: int = 9) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.randn(n_sensors, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(n_sensors, rank) * 0.02)
        self.gate = nn.Linear(cond_dim, n_sensors)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # h: (B, L, N, D); cond: (B, cond_dim)
        M = self.U @ self.V.T                                    # (N, N)
        delta = torch.einsum("nm,blmd->blnd", M, h)              # (B, L, N, D)
        g = torch.tanh(self.gate(cond))                          # (B, N)
        return h + g.view(h.shape[0], 1, h.shape[2], 1) * delta


class DegCandidateModel(CandidateModel):
    """CandidateModel + optional Module B / Module C insertion points.

    encode() below is a verbatim copy of CandidateModel.encode with exactly
    two marked changes:
      [INS-B] per-window attention bias inside the temporal transformer
      [INS-C] sensor mixer between the transformer and sensor pooling
    With both modules at zero-init the model is numerically identical to
    the stock PE backbone at initialisation.
    """

    def __init__(self, module_b: bool = True, module_c: bool = True,
                 cond_dim: int = 9, **kwargs) -> None:
        super().__init__(**kwargs)
        self.module_b = bool(module_b)
        self.module_c = bool(module_c)
        # M1p context injection (identical to PriorWrapper N_IN=9): every
        # module arm keeps the individualised prior on the pooled context,
        # so arms differ ONLY in the module-level condition.
        self.prior_proj = nn.Linear(cond_dim, self.d_model)
        nn.init.zeros_(self.prior_proj.weight)
        nn.init.zeros_(self.prior_proj.bias)
        if self.module_b:
            self.deg_bias = DegAttentionBias(cond_dim, self.layers[0].self_attn.num_heads,
                                             self.window_size)
            # replace the stock layer(s) with the deg-aware equivalents;
            # state-dict keys match nn.TransformerEncoderLayer exactly.
            self.layers = nn.ModuleList([
                DegAttnEncoderLayer(self.d_model,
                                    self.layers[0].self_attn.num_heads,
                                    2 * self.d_model, self.dropout.p)
                for _ in self.layers
            ])
        if self.module_c:
            self.mixer = DegSensorMixer(self.num_sensors, rank=4, cond_dim=cond_dim)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
                extras: Optional[torch.Tensor] = None,
                deg: Optional[torch.Tensor] = None,
                prior: Optional[torch.Tensor] = None):
        pooled = self.encode(x, cond=cond, extras=extras, deg=deg)
        context = pooled.mean(dim=1)
        if self.prior_proj is not None:
            context = context + self.prior_proj(prior)
        return {"prediction": self.reg_head(context), "embedding": context}

    def encode(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
               extras: Optional[torch.Tensor] = None,
               deg: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (self.module_b or self.module_c) and deg is None:
            raise ValueError("deg condition vector required for module B/C")
        batch, length, sensors = x.shape
        if extras is not None:
            baseline = extras[:, :sensors].unsqueeze(1)
            cycle = extras[:, sensors: sensors + 1]
            evidence_feats = extras[:, sensors + 1: sensors + 7]
        else:
            baseline = x[:, :1, :]
            cycle = x.new_zeros(batch, 1)
            evidence_feats = x.new_zeros(batch, 6)

        feats = [x]
        if self.use_baseline:
            feats.append(x - baseline)
        if self.use_ema_features:
            feats.append(self._ema(x, 0.5))
        if cond is not None:
            cond_t = cond if cond.ndim == 3 else cond.unsqueeze(1).expand(-1, length, -1)
        else:
            cond_t = x.new_zeros(batch, length, 3)
        stacked = torch.stack(feats, dim=-1)
        h = self.sensor_in(stacked)  # (B, L, N, D)

        hh = h.permute(0, 2, 3, 1).reshape(batch * sensors, self.d_model, length)
        hh = F.gelu(self.time_conv(hh))
        conv_out = hh.reshape(batch, sensors, self.d_model, length).permute(0, 3, 1, 2)
        h = self.time_norm(h + self.dropout(conv_out))

        if self.use_graph:
            flat = h.reshape(batch * length, sensors, self.d_model)
            flat_cond = cond_t.reshape(batch * length, 1, 3)
            flat = self.graph(flat, flat_cond)
            h = flat.reshape(batch, length, sensors, self.d_model)

        # shared temporal transformer across sensors
        tokens = h.permute(0, 2, 1, 3).reshape(batch * sensors, length, self.d_model)
        tokens = tokens + self.temporal_pe
        if self.module_b:
            bias = self.deg_bias(deg)                        # (B, L, L)
            bias = bias.repeat_interleave(sensors, dim=0)    # (B*N, L, L), b*N+n order
            for layer in self.layers:
                tokens = layer(tokens, bias)
        else:
            for layer in self.layers:
                tokens = layer(tokens)
        tokens = tokens.reshape(batch, sensors, length, self.d_model).permute(0, 2, 1, 3)

        # [INS-C] degradation-aware sensor mixer (before sensor pooling)
        if self.module_c:
            tokens = self.mixer(tokens, deg)

        score = torch.softmax(self.sensor_score(tokens).squeeze(-1), dim=-1)
        pooled = (score.unsqueeze(-1) * tokens).sum(dim=2)
        if self.use_cycle and self.use_pe:
            c_abs = cycle * 250.0
            ang = (2.0 * math.pi) * c_abs / self.pe_periods.view(1, -1)
            pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
            pooled = pooled + self.pe_proj(pe).unsqueeze(1)
        pooled = self.norm(pooled)
        return pooled
