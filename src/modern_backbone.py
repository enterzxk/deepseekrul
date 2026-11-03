"""Modern backbone for the G2 gate (spec section 3).

V4 = patchified input + M2 temporal mixer (RoPE + SwiGLU pre-norm block),
keeping V0's sensor readout, cycle PE and regression head verbatim.

Design rules from the frozen spec:
  * patch_len 5, stride 5 -> 10 tokens per sensor; Linear(15 -> 64)
    replaces the per-step 3 -> 64 mapping and the depthwise time conv.
  * M2: pre-norm Transformer block with RoPE (no learned temporal PE) and
    a SwiGLU feed-forward.
  * parameter budget within +-5% of the PE V0 body (42,306 parameters).
  * M3 (Mamba-2) is skipped: no mamba_ssm in the environment, and hand
    writing a slow scan is worse than not doing it.

Nothing here reads the test split.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

V0_PARAMS = 42_306          # PE V0 body (num_layers=1), recorded in metrics
PATCH_LEN = 5


def rope_cache(seq_len: int, d_head: int, device, base: float = 10000.0):
    """cos/sin tables of shape (S, d_head // 2)."""
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, S, H, Dh); cos/sin: (S, Dh // 2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.view(1, -1, 1, half)
    sin = sin.view(1, -1, 1, half)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RopeSwiGLUBlock(nn.Module):
    """Pre-norm block: RoPE attention + SwiGLU feed-forward."""

    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.heads = int(num_heads)
        self.d_head = d_model // self.heads
        self.n1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, ffn_hidden)
        self.w3 = nn.Linear(d_model, ffn_hidden)
        self.w2 = nn.Linear(ffn_hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        b, s, d = tokens.shape
        h = self.n1(tokens)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, s, self.heads, self.d_head)
        k = k.view(b, s, self.heads, self.d_head)
        v = v.view(b, s, self.heads, self.d_head)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))       # (B, H, S, Dh)
        att = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        att = att.transpose(1, 2).reshape(b, s, d)
        tokens = tokens + self.drop(self.proj(att))
        h = self.n2(tokens)
        ffn = self.w2(F.silu(self.w1(h)) * self.w3(h))
        return tokens + self.drop(ffn)


class PatchRopeModel(nn.Module):
    """V4: patch encoder + RoPE/SwiGLU mixer; V0 readout unchanged.

    Interface-compatible with CandidateModel (forward(x, cond, extras) ->
    dict with prediction / trajectory / embedding), so run scripts can swap
    the backbone without touching the training loop.
    """

    def __init__(
        self,
        num_sensors: int = 14,
        window_size: int = 50,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.15,
        use_baseline: bool = True,
        use_ema_features: bool = True,
        use_cycle: bool = True,
        use_pe: bool = True,
        ffn_hidden: int = 90,
        patch_len: int = PATCH_LEN,
        max_rul: float = 125.0,
    ) -> None:
        super().__init__()
        if not use_cycle:
            # the cycle PE is part of the kept V0 readout; a patch model
            # without it would no longer be "V4 of the same protocol".
            raise ValueError("PatchRopeModel keeps the frozen cycle PE (use_cycle=True)")
        self.num_sensors = int(num_sensors)
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.n_patches = window_size // self.patch_len
        if window_size % self.patch_len:
            raise ValueError("window_size must be divisible by patch_len")
        per_sensor = 1 + int(use_baseline) + int(use_ema_features)
        self.per_sensor = per_sensor

        self.patch_proj = nn.Linear(per_sensor * self.patch_len, d_model)
        self.blocks = nn.ModuleList(
            RopeSwiGLUBlock(d_model, num_heads, ffn_hidden, dropout)
            for _ in range(int(num_layers))
        )
        # kept from V0: bounded 4-period sinusoidal cycle PE
        self.register_buffer(
            "pe_periods", torch.tensor([25.0, 50.0, 125.0, 250.0]), persistent=False
        )
        self.pe_proj = nn.Linear(8, d_model)
        self.sensor_score = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
        nn.init.constant_(self.reg_head[-1].bias, 2.0)

    @staticmethod
    def _ema(x: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
        out = torch.empty_like(x)
        acc = x[:, 0]
        out[:, 0] = acc
        for t in range(1, x.shape[1]):
            acc = beta * acc + (1.0 - beta) * x[:, t]
            out[:, t] = acc
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        extras: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        b, length, sensors = x.shape
        if extras is not None:
            baseline = extras[:, :sensors].unsqueeze(1)
            cycle = extras[:, sensors: sensors + 1]
        else:
            baseline = x[:, :1, :]
            cycle = x.new_zeros(b, 1)

        feats = [x]
        if self.per_sensor >= 2:
            feats.append(x - baseline)
        if self.per_sensor >= 3:
            feats.append(self._ema(x, 0.5))
        feats = torch.stack(feats, dim=-1)                       # (B, L, N, F)

        # patchify: (B, L, N, F) -> (B, N, P, F*patch_len)
        f = feats.permute(0, 2, 1, 3).reshape(
            b, sensors, self.n_patches, -1)
        tokens = self.patch_proj(f)                              # (B, N, P, D)
        tokens = tokens.permute(0, 2, 1, 3).reshape(b * sensors, self.n_patches, self.d_model)

        cos, sin = rope_cache(self.n_patches, self.d_model // 4, x.device)
        for blk in self.blocks:
            tokens = blk(tokens, cos, sin)

        tokens = tokens.reshape(b, sensors, self.n_patches, self.d_model).permute(0, 2, 1, 3)
        # sensor readout over sensors dim (V0-style, at every patch position)
        score = torch.softmax(self.sensor_score(tokens).squeeze(-1), dim=-1)
        pooled = (score.unsqueeze(-1) * tokens).sum(dim=2)       # (B, P, D)

        c_abs = (cycle * 250.0).clamp(min=0.0)
        ang = (2.0 * math.pi) * c_abs / self.pe_periods.view(1, -1)
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        pooled = pooled + self.pe_proj(pe).unsqueeze(1)
        pooled = self.norm(pooled)

        context = pooled.mean(dim=1)
        prediction = self.reg_head(context)
        raw_traj = self.reg_head(pooled).squeeze(-1)
        return {"prediction": prediction, "trajectory": raw_traj, "embedding": context}
