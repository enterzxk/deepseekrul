"""Vanilla Transformer base with switchable MS / SA / HI modules.

BASE (all flags off) is a standard Transformer encoder:
  Linear(14->D) -> sinusoidal PE -> 2x TransformerEncoderLayer -> masked mean pool -> reg head

MS: multi-scale depthwise temporal conv (k=3/5/7) as a residual branch on the token sequence.
SA: per-sensor projection + cross-sensor Transformer layer, aggregated and added residually.
HI: training-only auxiliary head + monotonic penalty. No HI input.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SPAN = 200


def masked_mean(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    den = m.sum(dim=1).clamp_min(1.0)
    return (h * m).sum(dim=1) / den


class SinusoidalPE(nn.Module):
    """Standard fixed sinusoidal positional encoding (no learned table, no cycle input)."""
    def __init__(self, d_model, max_len=MAX_SPAN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, h):
        return h + self.pe[:, :h.size(1)]


class MSBranch(nn.Module):
    """Multi-scale depthwise temporal convolution, residual on the token sequence."""
    def __init__(self, d_model):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model)
                                    for k in (3, 5, 7)])
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, mask):
        x = h.transpose(1, 2)                       # (B,D,L)
        ms = sum(F.gelu(c(x)) for c in self.convs) / 3.0
        out = self.proj(ms.transpose(1, 2))
        return self.norm(h + out)


class SABranch(nn.Module):
    """Cross-sensor Transformer branch: per-sensor embedding -> attention across sensors -> aggregate."""
    def __init__(self, d_model, num_sensors, nhead=2, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.sensor_embed = nn.Linear(1, d_model)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.agg = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, h, mask):
        B, L, N = x.shape
        e = self.sensor_embed(x.unsqueeze(-1))       # (B,L,N,D)
        st = e.reshape(B * L, N, self.d_model)
        st = self.layer(st)
        st = st.reshape(B, L, N, self.d_model)
        pooled = st.mean(dim=2)                      # (B,L,D)
        return self.norm(h + self.agg(pooled))


class DSPTVanilla(nn.Module):
    def __init__(self, num_sensors=14, d_model=64, num_layers=2, dropout=0.2,
                 max_span=MAX_SPAN, use_ms=False, use_sa=False, use_hi=False):
        super().__init__()
        self.num_sensors = num_sensors
        self.d_model = d_model
        self.use_ms = use_ms
        self.use_sa = use_sa
        self.use_hi = use_hi
        self.embed = nn.Linear(num_sensors, d_model)
        self.pe = SinusoidalPE(d_model, max_span)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=2 * d_model,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.ms = MSBranch(d_model) if use_ms else None
        self.sa = SABranch(d_model, num_sensors, dropout=dropout) if use_sa else None
        self.hi_head = nn.Linear(d_model, 1) if use_hi else None
        self.reg_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, x, mask=None, baseline=None, cycle=None, mode='dual'):
        B, L, N = x.shape
        if mask is None:
            mask = torch.ones(B, L, device=x.device)
        h = self.embed(x)
        h = self.pe(h)
        if self.use_ms:
            h = self.ms(h, mask)
        if self.use_sa:
            h = self.sa(x, h, mask)
        pad = ~mask.bool()
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=pad)
        h = self.norm(h)
        pooled = masked_mean(h, mask)
        y = self.reg_head(pooled)
        out = {'prediction': y, 'y_A': y, 'y_B': y, 'embedding': pooled}
        if self.use_hi:
            out['hi'] = self.hi_head(pooled).reshape(-1)
            out['hi_seq'] = self.hi_head(h).squeeze(-1)
        return out


def build(num_sensors=14, d_model=64, num_layers=2, dropout=0.2, max_span=MAX_SPAN,
          use_ms=False, use_sa=False, use_hi=False):
    return DSPTVanilla(num_sensors=num_sensors, d_model=d_model, num_layers=num_layers,
                       dropout=dropout, max_span=max_span, use_ms=use_ms, use_sa=use_sa, use_hi=use_hi)
