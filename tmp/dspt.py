"""DSPT-v2: Dual-Span Pooled Transformer.

One shared encoder runs once over the full 100-step history; two nested suffix
pools read it (last 50 steps, and all valid steps) and a zero-parameter gate
mixes them. Parameter count equals the single-stream baseline of the same
temporal_pe width, so any gain cannot come from capacity.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SPAN = 100
SHORT_SPAN = 50


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """h (B,L,D), mask (B,L) with 1 = valid. Mean over valid steps only."""
    m = mask.unsqueeze(-1).to(h.dtype)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (h * m).sum(dim=1) / denom


class DSPT(nn.Module):
    def __init__(self, num_sensors=14, d_model=64, num_layers=2, dropout=0.2,
                 pe_periods=(25.0, 50.0, 125.0, 250.0), max_span=MAX_SPAN):
        super().__init__()
        self.num_sensors = num_sensors
        self.d_model = d_model
        self.max_span = max_span
        self.sensor_in = nn.Linear(3, d_model)
        self.time_conv = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.time_norm = nn.LayerNorm(d_model)
        self.register_buffer('pe_periods', torch.tensor(pe_periods))
        self.pe_proj = nn.Linear(2 * len(pe_periods), d_model)
        # learned positional table indexed by DISTANCE FROM THE END (0 = newest)
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_span, d_model))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=2 * d_model,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
        self.sensor_score = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.reg_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def _features(self, x, baseline, mask):
        """[raw, raw-baseline, causal EMA]; EMA recurses only over valid steps."""
        B, L, N = x.shape
        dev = x - baseline.unsqueeze(1)
        ema = x[:, 0].clone()
        out = []
        for t in range(L):
            ema = torch.where(mask[:, t:t + 1].bool(), 0.5 * ema + 0.5 * x[:, t], ema)
            out.append(ema)
        trend = torch.stack(out, dim=1)
        return torch.stack([x, dev, trend], dim=-1)  # (B,L,N,3)

    def encode(self, x, baseline, mask, cycle):
        B, L, N = x.shape
        feats = self._features(x, baseline, mask)
        h = self.sensor_in(feats)                                  # (B,L,N,D)
        hh = h.permute(0, 2, 3, 1).reshape(B * N, self.d_model, L)
        hh = F.gelu(self.time_conv(hh))
        conv_out = hh.reshape(B, N, self.d_model, L).permute(0, 3, 1, 2)
        h = self.time_norm(h + conv_out)                            # (B,L,N,D)
        tokens = h.permute(0, 2, 1, 3).reshape(B * N, L, self.d_model)
        # distance-from-end positional table
        dist = torch.arange(L - 1, -1, -1, device=x.device)          # 0 = newest
        tokens = tokens + self.temporal_pe[:, dist]
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = tokens.reshape(B, N, L, self.d_model).permute(0, 2, 1, 3)
        score = torch.softmax(self.sensor_score(tokens).squeeze(-1), dim=-1)  # (B,L,N)
        pooled = (score.unsqueeze(-1) * tokens).sum(dim=2)           # (B,L,D)
        c_abs = cycle * 250.0
        ang = (2.0 * math.pi) * c_abs.unsqueeze(1) / self.pe_periods.view(1, -1)
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        pooled = pooled + self.pe_proj(pe).unsqueeze(1)
        return self.norm(pooled)                                     # (B,L,D)

    def forward(self, x, mask=None, baseline=None, cycle=None):
        B, L, N = x.shape
        if mask is None:
            mask = torch.ones(B, L, device=x.device)
        if baseline is None:
            baseline = x[:, 0]
        if cycle is None:
            cycle = torch.zeros(B, device=x.device)
        h = self.encode(x, baseline, mask, cycle)                    # (B,L,D)
        # distance-from-end index for each position
        dist = torch.arange(L - 1, -1, -1, device=x.device)          # (L,)
        valid = mask.bool()
        short = valid & (dist.unsqueeze(0) < SHORT_SPAN)
        h_A = masked_mean(h, short.to(h.dtype))
        h_B = masked_mean(h, valid.to(h.dtype))
        m = mask.sum(dim=1, keepdim=True)
        g = torch.clamp((m - SHORT_SPAN) / SHORT_SPAN, 0.0, 1.0)
        h_f = g * h_B + (1.0 - g) * h_A
        y = self.reg_head(h_f)
        return {'prediction': y,
                'y_A': self.reg_head(h_A),
                'y_B': self.reg_head(h_B),
                'gate': g,
                'embedding': h_f}


def build(num_sensors=14, d_model=64, num_layers=2, dropout=0.2):
    return DSPT(num_sensors=num_sensors, d_model=d_model, num_layers=num_layers, dropout=dropout)
